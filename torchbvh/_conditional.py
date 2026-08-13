"""Conditional, one-branch-per-query batched MLS interpolation."""

import torch

from . import _C
from ._constants import SUPPORTED_DIMS
from ._handles import _batched_bvh_data, _temporary_bvh
from ._mls import (
    _MLS_FLATTENED_CHUNK_SIZE,
    _linear_mls_batched_head_banked_indexed_chunked_fused_forward,
)
from ._query import build_bvh_batched, _gather_batched_neighbor_positions
from ._validation import _as_contiguous, _validate_supported_k


_OP = "conditional_mls_interpolate"


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


def _query_knn_routed_batched(
    true_bvh,
    false_bvh,
    queries: torch.Tensor,
    routes: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Private exact routed k-NN over two fixed-size batched point BVHs."""
    true_data = _batched_bvh_data(true_bvh, _OP)
    false_data = _batched_bvh_data(false_bvh, _OP)
    query_order = _C.morton_sort_routed_queries_batched(
        queries,
        routes,
        true_data["scene_min"],
        true_data["scene_max"],
        false_data["scene_min"],
        false_data["scene_max"],
    )
    return _C.query_knn_routed_batched_ordered(
        true_data["node_aabbs"],
        true_data["sorted_indices"],
        false_data["node_aabbs"],
        false_data["sorted_indices"],
        queries,
        routes,
        query_order.contiguous(),
        true_data["num_leaves"],
        true_data["num_real_nodes"],
        true_data["leaf_level"],
        false_data["num_leaves"],
        false_data["num_real_nodes"],
        false_data["leaf_level"],
        true_data["dim"],
        k,
    )


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
    B, M, H, D, _ = _validate_inputs(
        mask, true_points, true_queries, true_features,
        false_points, false_queries, false_features, k,
    )
    mask = _as_contiguous(mask)
    true_points = _as_contiguous(true_points)
    false_points = _as_contiguous(false_points)
    true_queries = _as_contiguous(true_queries)
    false_queries = _as_contiguous(false_queries)
    true_features = _as_contiguous(true_features)
    false_features = _as_contiguous(false_features)

    route_by_head = mask.permute(0, 2, 1).unsqueeze(-1)
    selected_by_head = torch.where(
        route_by_head,
        true_queries.permute(0, 2, 1, 3),
        false_queries.permute(0, 2, 1, 3),
    ).contiguous()
    flat_queries = selected_by_head.detach().reshape(B, H * M, D).contiguous()
    flat_routes = mask.permute(0, 2, 1).reshape(B, H * M).contiguous()
    true_points_detached = true_points.detach().contiguous()
    false_points_detached = false_points.detach().contiguous()

    with _temporary_bvh(build_bvh_batched, true_points_detached) as true_bvh:
        with _temporary_bvh(build_bvh_batched, false_points_detached) as false_bvh:
            indices, squared_distances = _query_knn_routed_batched(
                true_bvh, false_bvh, flat_queries, flat_routes, k)

    combined_points = torch.cat((true_points_detached, false_points_detached), dim=1)
    neighbor_positions = _gather_batched_neighbor_positions(
        combined_points, indices, H * M, D)
    combined_features = torch.cat((true_features, false_features), dim=1)
    return _linear_mls_batched_head_banked_indexed_chunked_fused_forward(
        selected_by_head,
        neighbor_positions,
        indices,
        squared_distances,
        combined_features,
        return_grad=return_grad,
        chunk_size=4 * _MLS_FLATTENED_CHUNK_SIZE,
    )


__all__ = ["conditional_mls_interpolate"]
