"""Conditional, one-branch-per-query batched MLS interpolation."""

import torch

from ._extension import _C
from ._constants import SUPPORTED_DIMS
from ._handles import _batched_bvh_data, _temporary_bvh
from ._geometry import PointGeometry, _source_bvh, _validate_geometry
from ._mls import _linear_mls_spatial_indexed
from ._mls_routed import _routed_mls_spatial_indexed
from . import _mls_routed
from . import _mls_geometry
from ._conditional_layout import _pack_routed_queries
from ._query import _build_bvh_batched
from ._validation import _as_contiguous, _validate_supported_k


_OP = "conditional_mls_interpolate"
_PREPARED_KNN_THREADS = 0  # Experimental dispatch; promote after paired gates.
_DIRECT_FEATURE_BANKS = False
_BIN_RESOLUTION = 0
_BIN_DIAGNOSTICS = None
_ADAPTIVE_BINS = False
_FUSED_QUERY_PACK = False
_EXPERIMENTAL_FAST_POLICY = False
_FAST_POLICY_SPATIAL_BITS = 6


def _validate_inputs(
    mask: torch.Tensor,
    true_points: torch.Tensor,
    true_queries: torch.Tensor,
    true_features: torch.Tensor,
    false_points: torch.Tensor,
    false_queries: torch.Tensor,
    false_features: torch.Tensor,
    k: int,
) -> tuple[int, int, int, int, int]:
    _validate_supported_k(_OP, k)
    if mask.dim() != 3:
        raise ValueError(f"{_OP}: mask must have shape (B, M, H)")
    if mask.dtype != torch.bool:
        raise ValueError(f"{_OP}: mask must be bool")
    if not mask.is_cuda:
        raise ValueError(f"{_OP}: mask must be a CUDA tensor")

    for name, tensor in (("true_points", true_points), ("false_points", false_points)):
        if tensor.dim() != 3:
            raise ValueError(f"{_OP}: {name} must have shape (B, N_branch, D)")
    for name, tensor in (("true_queries", true_queries), ("false_queries", false_queries)):
        if tensor.dim() != 4:
            raise ValueError(f"{_OP}: {name} must have shape (B, M, H, D)")
    for name, tensor in (("true_features", true_features), ("false_features", false_features)):
        if tensor.dim() != 4:
            raise ValueError(f"{_OP}: {name} must have shape (B, N_branch, H, C)")

    B, M, H = mask.shape
    D = int(true_points.size(2))
    C = int(true_features.size(3))
    if B < 1 or M < 1 or H < 1 or C < 1:
        raise ValueError(f"{_OP}: B, M, H, and C must be positive")
    if D not in SUPPORTED_DIMS:
        raise ValueError(f"{_OP}: D must be 2 or 3")
    if true_points.size(1) < k or false_points.size(1) < k:
        raise ValueError(f"{_OP}: both point branches must contain at least k rows")

    expected_queries = (B, M, H, D)
    if tuple(true_queries.shape) != expected_queries or tuple(false_queries.shape) != expected_queries:
        raise ValueError(f"{_OP}: query branches must both have shape (B, M, H, D)")
    if true_points.size(0) != B or false_points.size(0) != B or false_points.size(2) != D:
        raise ValueError(f"{_OP}: point branches must share B and D")
    if tuple(true_features.shape[:3]) != (B, true_points.size(1), H):
        raise ValueError(f"{_OP}: true_features must match true_points and mask heads")
    if tuple(false_features.shape[:3]) != (B, false_points.size(1), H):
        raise ValueError(f"{_OP}: false_features must match false_points and mask heads")
    if false_features.size(3) != C:
        raise ValueError(f"{_OP}: feature branches must share C")

    tensors = (
        true_points, true_queries, true_features,
        false_points, false_queries, false_features,
    )
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise ValueError(f"{_OP}: point, query, and feature inputs must be float32")
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError(f"{_OP}: all inputs must be CUDA tensors")
    if any(tensor.device != mask.device for tensor in tensors):
        raise ValueError(f"{_OP}: all inputs must share a device")
    return B, M, H, D, C


def conditional_mls_interpolate(
    mask: torch.Tensor,
    *,
    true_points: torch.Tensor,
    true_queries: torch.Tensor,
    true_features: torch.Tensor,
    false_points: torch.Tensor,
    false_queries: torch.Tensor,
    false_features: torch.Tensor,
    k: int = 4,
    return_grad: bool = False,
    true_geometry: PointGeometry | None = None,
    false_geometry: PointGeometry | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Evaluate exactly one of two local-linear MLS branches per query.

    ``mask[b, m, h]`` selects the true branch. Inactive query rows may contain
    NaNs because they are neither selected nor traversed. BVH construction,
    source positions, Morton ordering, and neighbour selection are detached;
    gradients flow only to selected queries and active source features.

    Args:
        mask: Boolean route tensor of shape ``(B, M, H)``.
        true_points: True-branch source positions, ``(B, N_true, D)``.
        true_queries: True-branch queries, ``(B, M, H, D)``.
        true_features: True-branch feature bank, ``(B, N_true, H, C)``.
        false_points: False-branch source positions, ``(B, N_false, D)``.
        false_queries: False-branch queries, ``(B, M, H, D)``.
        false_features: False-branch feature bank, ``(B, N_false, H, C)``.
        k: Neighbour count; one of 4, 8, or 16.
        return_grad: Also return the spatial field gradient when true.

    Returns:
        Values with shape ``(B, M, H, C)``. If ``return_grad=True``, returns
        ``(values, field_gradient)`` with gradient shape ``(B, M, H, D, C)``.
    """
    B, M, H, D, C = _validate_inputs(
        mask, true_points, true_queries, true_features,
        false_points, false_queries, false_features, k,
    )
    _validate_geometry(true_points, true_geometry)
    _validate_geometry(false_points, false_geometry)
    fast_policy = (
        D == 2 and k == 4 and C in (4, 8, 16)
        and false_points.size(1) >= 512 and B * M * H >= 4096
    )
    mask = _as_contiguous(mask)
    true_points = _as_contiguous(true_points)
    false_points = _as_contiguous(false_points)
    true_queries = _as_contiguous(true_queries)
    false_queries = _as_contiguous(false_queries)
    true_features = _as_contiguous(true_features)
    false_features = _as_contiguous(false_features)

    if _FUSED_QUERY_PACK or fast_policy:
        selected_by_head, flat_routes = _pack_routed_queries(true_queries, false_queries, mask)
    else:
        route_by_head = mask.permute(0, 2, 1).unsqueeze(-1)
        selected_by_head = torch.where(
            route_by_head,
            true_queries.permute(0, 2, 1, 3),
            false_queries.permute(0, 2, 1, 3),
        ).contiguous()
        flat_routes = mask.permute(0, 2, 1).reshape(B, H * M).contiguous()
    flat_queries = selected_by_head.detach().reshape(B, H * M, D).contiguous()
    true_points_detached = true_points.detach().contiguous() if true_geometry is None else true_geometry._points
    false_points_detached = false_points.detach().contiguous() if false_geometry is None else false_geometry._points

    with _source_bvh(_build_bvh_batched, true_points_detached, true_geometry) as true_bvh:
        with _source_bvh(_build_bvh_batched, false_points_detached, false_geometry) as false_bvh:
            true_data = _batched_bvh_data(true_bvh, _OP)
            false_data = _batched_bvh_data(false_bvh, _OP)
            ordering_fn = _C.morton_sort_routed_queries_batched
            ordering_args = ()
            if fast_policy:
                ordering_fn = _C.morton_sort_routed_queries_narrow
                ordering_args = (_FAST_POLICY_SPATIAL_BITS,)
            if fast_policy and _FAST_POLICY_SPATIAL_BITS == 0:
                query_order = torch.arange(H * M, device=flat_queries.device).expand(B, -1).contiguous()
            else:
                query_order = ordering_fn(
                    flat_queries,
                    flat_routes,
                    true_data["scene_min"],
                    true_data["scene_max"],
                    false_data["scene_min"],
                    false_data["scene_max"],
                    *ordering_args,
                ).contiguous()
            query_fn = _C.query_knn_routed_batched_cached_bounds_spatial
            traversal_args = ()
            if _PREPARED_KNN_THREADS and true_geometry is not None and false_geometry is not None:
                query_fn = _C.query_knn_routed_explicit
                traversal_args = tuple(data[key] for data in (true_data, false_data)
                                       for key in ("left_child_mem", "right_child_mem", "mem_to_leaf")) + (_PREPARED_KNN_THREADS,)
            if (_BIN_RESOLUTION or fast_policy) and D == 2 and k == 4 and false_points.size(1) >= 512:
                resolution = _BIN_RESOLUTION
                if fast_policy:
                    # Target average occupancy <=4 up to the resolution cap. A fixed
                    # 64x64 grid exhausts the exact search's 64-candidate budget
                    # when dense queries need to visit adjacent cells.
                    resolution = 16
                    while resolution < 256 and false_points.size(1) > 4 * resolution * resolution:
                        resolution *= 2
                elif _ADAPTIVE_BINS:
                    resolution = min(resolution, 64 if false_points.size(1) >= 8192 else
                                     32 if false_points.size(1) >= 2048 else 16)
                bins = false_data.setdefault("point_bins", {})
                if resolution not in bins:
                    bins[resolution] = _C.build_point_bins(
                        false_points_detached, false_data["scene_min"], false_data["scene_max"], resolution)
                query_fn = _C.query_knn_routed_bins
                traversal_args = (false_points_detached, false_data["scene_min"], false_data["scene_max"],
                                  *bins[resolution], resolution)
            knn_result = (
                query_fn(
                    true_data["node_aabbs"],
                    true_data["sorted_indices"],
                    false_data["node_aabbs"],
                    false_data["sorted_indices"],
                    flat_queries,
                    flat_routes,
                    query_order,
                    true_data["num_leaves"],
                    true_data["num_real_nodes"],
                    false_data["num_leaves"],
                    false_data["num_real_nodes"],
                    D,
                    k,
                    *traversal_args,
                )
            )
            indices, squared_distances = knn_result[:2]
            if len(knn_result) == 3 and _BIN_DIAGNOSTICS is not None:
                _BIN_DIAGNOSTICS.append(knn_result[2])

    combined_points = torch.cat(
        (true_points_detached, false_points_detached), dim=1
    ).contiguous()
    source_count = combined_points.size(1)
    channels = true_features.size(-1)
    geometry_owner = _mls_geometry.eligible(D, k, channels, B * H * M)
    mls_fn = _linear_mls_spatial_indexed
    mls_kwargs = {}
    query_major_output = False
    if geometry_owner:
        mls_fn = _mls_geometry.interpolate
        feature_args = (true_features, false_features)
        query_major_output = True
        mls_kwargs["query_major_output"] = True
    elif (_DIRECT_FEATURE_BANKS or fast_policy) and channels <= 32:
        mls_fn = _routed_mls_spatial_indexed
        feature_args = (true_features, false_features)
        query_major_output = fast_policy or _mls_routed._QUERY_MAJOR_OUTPUT
        if query_major_output:
            mls_kwargs["query_major_output"] = True
    else:
        combined_features = torch.cat((true_features, false_features), dim=1)
        feature_args = (combined_features.permute(0, 2, 1, 3).reshape(
            B * H, source_count, channels).contiguous(),)
    result = mls_fn(
        selected_by_head.reshape(B * H * M, D),
        combined_points,
        indices.reshape(B * H * M, k),
        squared_distances.reshape(B * H * M, k),
        *feature_args,
        query_order.reshape(B * H * M),
        queries_per_batch=H * M,
        queries_per_head=M,
        return_grad=return_grad,
        **mls_kwargs,
    )
    if return_grad:
        values, gradient = result
        if query_major_output:
            return values.reshape(B, M, H, channels), gradient.reshape(B, M, H, D, channels)
        return (
            values.reshape(B, H, M, channels).permute(0, 2, 1, 3).contiguous(),
            gradient.reshape(B, H, M, D, channels)
            .permute(0, 2, 1, 3, 4).contiguous(),
        )
    if query_major_output:
        return result.reshape(B, M, H, channels)
    return result.reshape(B, H, M, channels).permute(0, 2, 1, 3).contiguous()


__all__ = ["conditional_mls_interpolate"]
