from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from . import _C
from ._constants import (
    EXACT_DISTANCE_EPSILON,
    MLS_BANDWIDTH_MIN,
    MLS_REGULARIZATION,
    SUPPORTED_DIMS,
)
from ._handles import _batched_bvh_data, _temporary_bvh
from ._query import _build_bvh_batched
from ._validation import _as_contiguous, _validate_supported_k


class _LinearMLSSpatialIndexed(torch.autograd.Function):
    """Packed MLS over Morton-ordered rows with positions gathered in-kernel."""

    @staticmethod
    def forward(
        ctx,
        displaced_points,
        source_points,
        indices,
        squared_distances,
        features,
        query_order,
        queries_per_batch,
        queries_per_head,
    ):
        ctx.set_materialize_grads(False)
        outputs = _C.mls_packed_indexed_forward(
            displaced_points.contiguous(),
            source_points.contiguous(),
            indices.contiguous(),
            squared_distances.contiguous(),
            features.contiguous(),
            query_order.contiguous(),
            int(queries_per_batch),
            int(queries_per_head),
            MLS_REGULARIZATION,
            MLS_BANDWIDTH_MIN,
            EXACT_DISTANCE_EPSILON,
        )
        interpolated, field_gradient, factors, exact_counts = outputs
        ctx.queries_per_batch = int(queries_per_batch)
        ctx.queries_per_head = int(queries_per_head)
        ctx.save_for_backward(
            displaced_points,
            source_points,
            indices,
            squared_distances,
            features,
            query_order,
            factors,
            exact_counts,
        )
        return interpolated, field_gradient

    @staticmethod
    @once_differentiable
    def backward(ctx, d_interpolated, d_field_gradient):
        (
            displaced,
            sources,
            indices,
            distances,
            features,
            order,
            factors,
            exact_counts,
        ) = ctx.saved_tensors
        if d_interpolated is None:
            d_interpolated = torch.zeros(
                (displaced.size(0), features.size(2)),
                device=features.device,
                dtype=features.dtype,
            )
        d_field_arg = (
            d_field_gradient.contiguous()
            if d_field_gradient is not None
            else torch.empty(0, device=features.device, dtype=features.dtype)
        )
        d_features, d_displaced = _C.mls_packed_indexed_backward(
            displaced.contiguous(),
            sources.contiguous(),
            indices.contiguous(),
            distances.contiguous(),
            features.contiguous(),
            order.contiguous(),
            factors.contiguous(),
            exact_counts.contiguous(),
            d_interpolated.contiguous(),
            d_field_arg,
            ctx.queries_per_batch,
            ctx.queries_per_head,
            MLS_BANDWIDTH_MIN,
            EXACT_DISTANCE_EPSILON,
        )
        return (
            d_displaced if ctx.needs_input_grad[0] else None,
            None,
            None,
            None,
            d_features if ctx.needs_input_grad[4] else None,
            None,
            None,
            None,
        )


def _linear_mls_spatial_indexed(
    displaced_points: torch.Tensor,
    source_points: torch.Tensor,
    indices: torch.Tensor,
    squared_distances: torch.Tensor,
    features: torch.Tensor,
    query_order: torch.Tensor,
    *,
    queries_per_batch: int,
    queries_per_head: int,
    return_grad: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    values, gradient = _LinearMLSSpatialIndexed.apply(
        displaced_points,
        source_points,
        indices,
        squared_distances,
        features,
        query_order,
        queries_per_batch,
        queries_per_head,
    )
    return (values, gradient) if return_grad else values


def _spatial_mls_batched_heads(
    points: torch.Tensor,
    displaced_points: torch.Tensor,
    features: torch.Tensor,
    k: int,
    *,
    return_grad: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Production Morton-ordered MLS with indexed source-position loads."""
    batch, source_count, dim = points.shape
    _, queries, heads, _ = displaced_points.shape
    channels = features.size(-1)
    displaced_by_head = displaced_points.permute(0, 2, 1, 3).contiguous()
    flat_queries = displaced_by_head.detach().reshape(
        batch, heads * queries, dim
    ).contiguous()
    detached_points = points.detach().contiguous()

    with _temporary_bvh(_build_bvh_batched, detached_points) as bvh:
        data = _batched_bvh_data(bvh, "bvh_mls_interpolate_batched_heads")
        query_order = _C.morton_sort_queries_narrow(
            flat_queries, data["scene_min"], data["scene_max"]
        ).contiguous()
        indices, squared_distances = _C.query_knn_batched_cached_bounds_spatial(
            data["node_aabbs"],
            data["sorted_indices"],
            flat_queries,
            query_order,
            data["num_leaves"],
            data["num_real_nodes"],
            dim,
            k,
        )

    feature_bank = features.permute(0, 2, 1, 3).reshape(
        batch * heads, source_count, channels
    ).contiguous()
    result = _linear_mls_spatial_indexed(
        displaced_by_head.reshape(batch * heads * queries, dim),
        detached_points,
        indices.reshape(batch * heads * queries, k),
        squared_distances.reshape(batch * heads * queries, k),
        feature_bank,
        query_order.reshape(batch * heads * queries),
        queries_per_batch=heads * queries,
        queries_per_head=queries,
        return_grad=return_grad,
    )
    if return_grad:
        values, gradient = result
        return (
            values.reshape(batch, heads, queries, channels)
            .permute(0, 2, 1, 3)
            .contiguous(),
            gradient.reshape(batch, heads, queries, dim, channels)
            .permute(0, 2, 1, 3, 4)
            .contiguous(),
        )
    return (
        result.reshape(batch, heads, queries, channels)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def _validate_mls_inputs(
    points: torch.Tensor,
    displaced_points: torch.Tensor,
    features: torch.Tensor,
    k: int,
) -> None:
    prefix = "mls_interpolate"
    _validate_supported_k(prefix, k)
    if points.dim() != 2:
        raise ValueError(f"{prefix}: points must have shape (N, D)")
    if displaced_points.dim() != 2:
        raise ValueError(f"{prefix}: displaced_points must have shape (M, D)")
    if features.dim() != 2:
        raise ValueError(f"{prefix}: features must have shape (N, C)")
    if points.size(1) not in SUPPORTED_DIMS:
        raise ValueError(f"{prefix}: D must be 2 or 3")
    if displaced_points.size(1) != points.size(1):
        raise ValueError(f"{prefix}: displaced_points second dimension must match points")
    if features.size(0) != points.size(0):
        raise ValueError(f"{prefix}: features first dimension must match points")
    if points.size(0) < k:
        raise ValueError(f"{prefix}: points must contain at least k rows")
    if points.device != displaced_points.device or points.device != features.device:
        raise ValueError(f"{prefix}: inputs must share a device")
    if any(tensor.dtype != torch.float32 for tensor in (points, displaced_points, features)):
        raise ValueError(f"{prefix}: inputs must be float32")
    if any(not tensor.is_cuda for tensor in (points, displaced_points, features)):
        raise ValueError(f"{prefix}: inputs must be CUDA tensors")


def _validate_batched_mls_inputs(
    points: torch.Tensor,
    displaced_points: torch.Tensor,
    features: torch.Tensor,
    k: int,
) -> None:
    prefix = "mls_interpolate"
    _validate_supported_k(prefix, k)
    if points.dim() != 3:
        raise ValueError(f"{prefix}: points must have shape (B, N, D)")
    if displaced_points.dim() != 3:
        raise ValueError(f"{prefix}: displaced_points must have shape (B, M, D)")
    if features.dim() != 3:
        raise ValueError(f"{prefix}: features must have shape (B, N, C)")
    if points.size(2) not in SUPPORTED_DIMS:
        raise ValueError(f"{prefix}: D must be 2 or 3")
    if displaced_points.size(0) != points.size(0):
        raise ValueError(f"{prefix}: displaced_points batch size must match points")
    if displaced_points.size(2) != points.size(2):
        raise ValueError(f"{prefix}: displaced_points last dimension must match points")
    if features.size(0) != points.size(0) or features.size(1) != points.size(1):
        raise ValueError(f"{prefix}: features first two dimensions must match points")
    if points.size(1) < k:
        raise ValueError(f"{prefix}: points must contain at least k rows per sample")
    if points.device != displaced_points.device or points.device != features.device:
        raise ValueError(f"{prefix}: inputs must share a device")
    if any(tensor.dtype != torch.float32 for tensor in (points, displaced_points, features)):
        raise ValueError(f"{prefix}: inputs must be float32")
    if any(not tensor.is_cuda for tensor in (points, displaced_points, features)):
        raise ValueError(f"{prefix}: inputs must be CUDA tensors")


def _mls_interpolate_single(
    points: torch.Tensor,
    displaced_points: torch.Tensor,
    features: torch.Tensor,
    k: int,
    *,
    return_grad: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    _validate_mls_inputs(points, displaced_points, features, k)
    result = _spatial_mls_batched_heads(
        _as_contiguous(points).unsqueeze(0),
        _as_contiguous(displaced_points).unsqueeze(0).unsqueeze(2),
        _as_contiguous(features).unsqueeze(0).unsqueeze(2),
        k,
        return_grad=return_grad,
    )
    if return_grad:
        values, gradient = result
        return values[0, :, 0], gradient[0, :, 0]
    return result[0, :, 0]


def _mls_interpolate_batched(
    points: torch.Tensor,
    displaced_points: torch.Tensor,
    features: torch.Tensor,
    k: int,
    *,
    return_grad: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    _validate_batched_mls_inputs(points, displaced_points, features, k)
    result = _spatial_mls_batched_heads(
        _as_contiguous(points),
        _as_contiguous(displaced_points).unsqueeze(2),
        _as_contiguous(features).unsqueeze(2),
        k,
        return_grad=return_grad,
    )
    if return_grad:
        values, gradient = result
        return values[:, :, 0], gradient[:, :, 0]
    return result[:, :, 0]


def bvh_mls_interpolate_batched_heads(
    points: torch.Tensor,
    displaced_points: torch.Tensor,
    features: torch.Tensor,
    k: int = 8,
    *,
    return_grad: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Batched matching-head local-linear MLS interpolation."""
    prefix = "bvh_mls_interpolate_batched_heads"
    if points.dim() != 3:
        raise ValueError(f"{prefix}: points must have shape (B, N, D)")
    if displaced_points.dim() != 4:
        raise ValueError(f"{prefix}: displaced_points must have shape (B, M, H, D)")
    if features.dim() != 4:
        raise ValueError(f"{prefix}: features must have shape (B, N, H, C)")

    batch, source_count, dim = points.shape
    query_batch, _, heads, query_dim = displaced_points.shape
    feature_batch, feature_count, feature_heads, channels = features.shape
    _validate_supported_k(prefix, k)
    if dim not in SUPPORTED_DIMS:
        raise ValueError(f"{prefix}: D must be 2 or 3")
    if query_batch != batch or feature_batch != batch:
        raise ValueError(f"{prefix}: batch dimensions must match")
    if query_dim != dim:
        raise ValueError(f"{prefix}: query dimension must match points")
    if feature_count != source_count:
        raise ValueError(f"{prefix}: features must match points")
    if feature_heads != heads:
        raise ValueError(f"{prefix}: feature and query heads must match")
    if source_count < k:
        raise ValueError(f"{prefix}: points must contain at least k rows")
    if channels < 1:
        raise ValueError(f"{prefix}: features must contain at least one channel")
    if points.device != displaced_points.device or points.device != features.device:
        raise ValueError(f"{prefix}: inputs must share a device")
    if any(tensor.dtype != torch.float32 for tensor in (points, displaced_points, features)):
        raise ValueError(f"{prefix}: inputs must be float32")
    if any(not tensor.is_cuda for tensor in (points, displaced_points, features)):
        raise ValueError(f"{prefix}: inputs must be CUDA tensors")

    return _spatial_mls_batched_heads(
        _as_contiguous(points),
        _as_contiguous(displaced_points),
        _as_contiguous(features),
        k,
        return_grad=return_grad,
    )


def mls_interpolate(
    points: torch.Tensor,
    displaced_points: torch.Tensor,
    features: torch.Tensor,
    k: int = 8,
    *,
    return_grad: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Interpolate features for a single or fixed-size batch of point clouds."""
    if points.dim() == 3:
        return _mls_interpolate_batched(
            points, displaced_points, features, k, return_grad=return_grad
        )
    return _mls_interpolate_single(
        points, displaced_points, features, k, return_grad=return_grad
    )


__all__ = ["bvh_mls_interpolate_batched_heads", "mls_interpolate"]
