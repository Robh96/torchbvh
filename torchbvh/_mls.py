
import torch

from . import _C
from ._constants import (
    EXACT_DISTANCE_EPSILON,
    MLS_BANDWIDTH_MIN,
    MLS_REGULARIZATION,
    SUPPORTED_DIMS,
)
from ._handles import _temporary_bvh
from ._query import build_bvh, build_bvh_batched, query_knn, query_knn_batched
from ._validation import _as_contiguous, _validate_supported_k


_MLS_FLATTENED_CHUNK_SIZE = 16384


class _LinearMLSFusedForward(torch.autograd.Function):
    """Private Stage 11 Candidate E fused CUDA MLS forward+backward.

    Forward returns (interpolated, field_gradient, factors, exact_counts).
    Backward uses implicit differentiation of the saved Cholesky factor;
    no gradient through BVH construction, indices, squared_distances, or
    neighbor_positions.
    """

    @staticmethod
    def forward(
        ctx,
        displaced_points: torch.Tensor,
        neighbor_positions: torch.Tensor,
        indices: torch.Tensor,
        squared_distances: torch.Tensor,
        features: torch.Tensor,
        feature_batch: torch.Tensor,
    ):
        ctx.mark_non_differentiable(feature_batch)
        interpolated, field_gradient, factors, exact_counts = _C.mls_fused_forward(
            displaced_points.contiguous(),
            neighbor_positions.contiguous(),
            indices.contiguous(),
            squared_distances.contiguous(),
            features.contiguous(),
            feature_batch.contiguous(),
            MLS_REGULARIZATION,
            MLS_BANDWIDTH_MIN,
            EXACT_DISTANCE_EPSILON,
        )
        ctx.save_for_backward(
            displaced_points,
            neighbor_positions,
            indices,
            squared_distances,
            features,
            feature_batch,
            factors,
            exact_counts,
        )
        return interpolated, field_gradient, factors, exact_counts

    @staticmethod
    def backward(ctx, d_interpolated, d_field_gradient, d_factors, d_exact_counts):
        (
            displaced_points,
            neighbor_positions,
            indices,
            squared_distances,
            features,
            feature_batch,
            factors,
            exact_counts,
        ) = ctx.saved_tensors

        needs_disp_grad = ctx.needs_input_grad[0]
        needs_feat_grad = ctx.needs_input_grad[4]

        if not needs_disp_grad and not needs_feat_grad:
            return None, None, None, None, None, None

        if d_interpolated is None:
            d_interpolated = torch.zeros(
                displaced_points.size(0), features.size(2),
                device=features.device, dtype=features.dtype,
            )

        # Pass empty tensor (numel==0) to signal "no upstream field_gradient".
        if d_field_gradient is not None:
            d_field_grad_arg = d_field_gradient.contiguous()
        else:
            d_field_grad_arg = torch.empty(0, device=features.device, dtype=features.dtype)

        d_features_out, d_displaced_out = _C.mls_fused_backward(
            displaced_points.contiguous(),
            neighbor_positions.contiguous(),
            indices.contiguous(),
            squared_distances.contiguous(),
            features.contiguous(),
            feature_batch.contiguous(),
            factors.contiguous(),
            exact_counts.contiguous(),
            d_interpolated.contiguous(),
            d_field_grad_arg,
            MLS_BANDWIDTH_MIN,
            EXACT_DISTANCE_EPSILON,
        )

        d_disp = d_displaced_out if needs_disp_grad else None
        d_feat = d_features_out if needs_feat_grad else None
        return d_disp, None, None, None, d_feat, None


def _linear_mls_fused_forward(
    displaced_points: torch.Tensor,
    neighbor_positions: torch.Tensor,
    indices: torch.Tensor,
    squared_distances: torch.Tensor,
    features: torch.Tensor,
    feature_batch: torch.Tensor | None = None,
    *,
    return_grad: bool,
    return_aux: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if displaced_points.dim() != 2:
        raise ValueError("_linear_mls_fused_forward: displaced_points must have shape (M, D)")
    if neighbor_positions.dim() != 3:
        raise ValueError("_linear_mls_fused_forward: neighbor_positions must have shape (M, k, D)")
    if indices.dim() != 2 or squared_distances.dim() != 2:
        raise ValueError("_linear_mls_fused_forward: indices and squared_distances must have shape (M, k)")
    if features.dim() == 2:
        feature_bank = features.unsqueeze(0)
    elif features.dim() == 3:
        feature_bank = features
    else:
        raise ValueError("_linear_mls_fused_forward: features must have shape (N, C) or (B, N, C)")
    if feature_batch is None:
        feature_batch = torch.zeros(displaced_points.size(0), device=displaced_points.device, dtype=torch.int64)
    interpolated, field_gradient, factors, exact_counts = _LinearMLSFusedForward.apply(
        displaced_points,
        neighbor_positions,
        indices,
        squared_distances,
        feature_bank,
        feature_batch,
    )
    if return_aux:
        return interpolated, field_gradient, factors, exact_counts
    if return_grad:
        return interpolated, field_gradient
    return interpolated


class BVHQuery(torch.autograd.Function):
    """Non-differentiable boundary around BVH construction and k-NN selection."""

    @staticmethod
    def forward(ctx, points: torch.Tensor, displaced_points: torch.Tensor, k: int):
        points_detached = points.detach().contiguous()
        query_detached = displaced_points.detach().contiguous()
        with _temporary_bvh(build_bvh, points_detached) as bvh:
            indices, squared_distances, neighbor_positions = query_knn(
                bvh,
                query_detached,
                k,
                source_points=points_detached,
                sort_queries=True,
            )
        return indices.detach(), squared_distances.detach(), neighbor_positions.detach()

    @staticmethod
    def backward(ctx, grad_indices, grad_squared_distances, grad_neighbor_positions):
        return None, None, None


class BatchedBVHQuery(torch.autograd.Function):
    """Non-differentiable boundary around batched BVH construction and k-NN selection."""

    @staticmethod
    def forward(ctx, points: torch.Tensor, displaced_points: torch.Tensor, k: int):
        points_detached = points.detach().contiguous()
        query_detached = displaced_points.detach().contiguous()
        with _temporary_bvh(build_bvh_batched, points_detached) as bvh:
            indices, squared_distances, neighbor_positions = query_knn_batched(
                bvh,
                query_detached,
                k,
                source_points=points_detached,
                sort_queries=True,
            )
        return indices.detach(), squared_distances.detach(), neighbor_positions.detach()

    @staticmethod
    def backward(ctx, grad_indices, grad_squared_distances, grad_neighbor_positions):
        return None, None, None


def _validate_mls_inputs(
    points: torch.Tensor,
    displaced_points: torch.Tensor,
    features: torch.Tensor,
    k: int,
) -> None:
    _validate_supported_k("bvh_mls_interpolate", k)
    if points.dim() != 2:
        raise ValueError("bvh_mls_interpolate: points must have shape (N, D)")
    if displaced_points.dim() != 2:
        raise ValueError("bvh_mls_interpolate: displaced_points must have shape (N_queries, D)")
    if features.dim() != 2:
        raise ValueError("bvh_mls_interpolate: features must have shape (N, F)")
    if points.size(1) not in SUPPORTED_DIMS:
        raise ValueError("bvh_mls_interpolate: D must be 2 or 3")
    if displaced_points.size(1) != points.size(1):
        raise ValueError("bvh_mls_interpolate: displaced_points second dimension must match points")
    if features.size(0) != points.size(0):
        raise ValueError("bvh_mls_interpolate: features first dimension must match points")
    if points.size(0) < k:
        raise ValueError("bvh_mls_interpolate: points must contain at least k rows")
    if points.device != displaced_points.device or points.device != features.device:
        raise ValueError("bvh_mls_interpolate: points, displaced_points, and features must be on the same device")
    if points.dtype != torch.float32 or displaced_points.dtype != torch.float32:
        raise ValueError("bvh_mls_interpolate: points and displaced_points must be float32")
    if features.dtype != torch.float32:
        raise ValueError("bvh_mls_interpolate: features must be float32")
    if not points.is_cuda or not displaced_points.is_cuda or not features.is_cuda:
        raise ValueError("bvh_mls_interpolate: points, displaced_points, and features must be CUDA tensors")


def _validate_batched_mls_inputs(
    points: torch.Tensor,
    displaced_points: torch.Tensor,
    features: torch.Tensor,
    k: int,
) -> None:
    _validate_supported_k("bvh_mls_interpolate_batched", k)
    if points.dim() != 3:
        raise ValueError("bvh_mls_interpolate_batched: points must have shape (B, N, D)")
    if displaced_points.dim() != 3:
        raise ValueError("bvh_mls_interpolate_batched: displaced_points must have shape (B, M, D)")
    if features.dim() != 3:
        raise ValueError("bvh_mls_interpolate_batched: features must have shape (B, N, C)")
    if points.size(2) not in SUPPORTED_DIMS:
        raise ValueError("bvh_mls_interpolate_batched: D must be 2 or 3")
    if displaced_points.size(0) != points.size(0):
        raise ValueError("bvh_mls_interpolate_batched: displaced_points batch size must match points")
    if displaced_points.size(2) != points.size(2):
        raise ValueError("bvh_mls_interpolate_batched: displaced_points last dimension must match points")
    if features.size(0) != points.size(0) or features.size(1) != points.size(1):
        raise ValueError("bvh_mls_interpolate_batched: features first two dimensions must match points")
    if points.size(1) < k:
        raise ValueError("bvh_mls_interpolate_batched: points must contain at least k rows per sample")
    if points.device != displaced_points.device or points.device != features.device:
        raise ValueError(
            "bvh_mls_interpolate_batched: points, displaced_points, and features must be on the same device"
        )
    if points.dtype != torch.float32 or displaced_points.dtype != torch.float32:
        raise ValueError("bvh_mls_interpolate_batched: points and displaced_points must be float32")
    if features.dtype != torch.float32:
        raise ValueError("bvh_mls_interpolate_batched: features must be float32")
    if not points.is_cuda or not displaced_points.is_cuda or not features.is_cuda:
        raise ValueError("bvh_mls_interpolate_batched: points, displaced_points, and features must be CUDA tensors")


def _linear_mls_chunk(
    displaced_points: torch.Tensor,
    neighbor_positions: torch.Tensor,
    neighbor_features: torch.Tensor,
    squared_distances: torch.Tensor,
    *,
    return_grad: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    delta = displaced_points.unsqueeze(1) - neighbor_positions

    with torch.no_grad():
        k = squared_distances.size(-1)
        bandwidth = squared_distances[..., (k - 1) // 2 : (k - 1) // 2 + 1].clamp(
            min=MLS_BANDWIDTH_MIN
        )

    delta_sq = delta.square().sum(dim=-1)
    weights = torch.exp(-delta_sq / (2.0 * bandwidth)).unsqueeze(-1)
    ones = torch.ones((*delta.shape[:2], 1), device=delta.device, dtype=delta.dtype)
    basis = torch.cat((ones, delta), dim=-1)
    weighted_basis = weights * basis

    normal_matrix = torch.bmm(weighted_basis.transpose(1, 2), basis)
    rhs = torch.bmm(weighted_basis.transpose(1, 2), neighbor_features)
    eye = torch.eye(basis.size(-1), device=basis.device, dtype=basis.dtype).unsqueeze(0)
    coeffs = torch.linalg.solve(normal_matrix + MLS_REGULARIZATION * eye, rhs)

    interpolated = coeffs[:, 0, :]
    if return_grad:
        field_gradient = coeffs[:, 1:, :]

    exact_mask = squared_distances <= EXACT_DISTANCE_EPSILON
    if exact_mask.any():
        exact_weights = exact_mask.to(neighbor_features.dtype).unsqueeze(-1)
        exact_counts = exact_weights.sum(dim=1).clamp(min=1.0)
        exact_average = (neighbor_features * exact_weights).sum(dim=1) / exact_counts
        interpolated = torch.where(exact_mask.any(dim=1, keepdim=True), exact_average, interpolated)

    if return_grad:
        return interpolated, field_gradient
    return interpolated


def _linear_mls_indexed_chunked_fused_forward(
    displaced_points: torch.Tensor,
    neighbor_positions: torch.Tensor,
    indices: torch.Tensor,
    squared_distances: torch.Tensor,
    features: torch.Tensor,
    *,
    return_grad: bool,
    return_aux: bool = False,
    chunk_size: int = _MLS_FLATTENED_CHUNK_SIZE,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    interpolated_chunks = []
    field_gradient_chunks = [] if (return_grad or return_aux) else None
    factor_chunks = [] if return_aux else None
    exact_count_chunks = [] if return_aux else None

    total_queries = displaced_points.size(0)
    for start in range(0, total_queries, chunk_size):
        end = min(start + chunk_size, total_queries)
        result = _linear_mls_fused_forward(
            displaced_points[start:end],
            neighbor_positions[start:end],
            indices[start:end],
            squared_distances[start:end],
            features,
            return_grad=return_grad,
            return_aux=return_aux,
        )
        if return_aux:
            interpolated_chunk, field_gradient_chunk, factor_chunk, exact_count_chunk = result
            interpolated_chunks.append(interpolated_chunk)
            field_gradient_chunks.append(field_gradient_chunk)
            factor_chunks.append(factor_chunk)
            exact_count_chunks.append(exact_count_chunk)
        elif return_grad:
            interpolated_chunk, field_gradient_chunk = result
            interpolated_chunks.append(interpolated_chunk)
            field_gradient_chunks.append(field_gradient_chunk)
        else:
            interpolated_chunks.append(result)

    interpolated = torch.cat(interpolated_chunks, dim=0)
    if return_aux:
        return (
            interpolated,
            torch.cat(field_gradient_chunks, dim=0),
            torch.cat(factor_chunks, dim=0),
            torch.cat(exact_count_chunks, dim=0),
        )
    if return_grad:
        return interpolated, torch.cat(field_gradient_chunks, dim=0)
    return interpolated


def _linear_mls_batched_indexed_chunked_fused_forward(
    displaced_points: torch.Tensor,
    neighbor_positions: torch.Tensor,
    indices: torch.Tensor,
    squared_distances: torch.Tensor,
    features: torch.Tensor,
    *,
    return_grad: bool,
    return_aux: bool = False,
    chunk_size: int = _MLS_FLATTENED_CHUNK_SIZE,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, M, K, D = neighbor_positions.shape
    C = features.size(-1)

    flat_displaced_points = displaced_points.reshape(B * M, D)
    flat_neighbor_positions = neighbor_positions.reshape(B * M, K, D)
    flat_indices = indices.reshape(B * M, K)
    flat_squared_distances = squared_distances.reshape(B * M, K)
    flat_batch = torch.arange(B, device=features.device, dtype=torch.int64).unsqueeze(1).expand(B, M).reshape(B * M)
    flat_features = features.reshape(B, features.size(1), C)

    interpolated_chunks = []
    field_gradient_chunks = [] if (return_grad or return_aux) else None
    factor_chunks = [] if return_aux else None
    exact_count_chunks = [] if return_aux else None

    total_queries = flat_displaced_points.size(0)
    for start in range(0, total_queries, chunk_size):
        end = min(start + chunk_size, total_queries)
        result = _linear_mls_fused_forward(
            flat_displaced_points[start:end],
            flat_neighbor_positions[start:end],
            flat_indices[start:end],
            flat_squared_distances[start:end],
            flat_features,
            flat_batch[start:end],
            return_grad=return_grad,
            return_aux=return_aux,
        )
        if return_aux:
            interpolated_chunk, field_gradient_chunk, factor_chunk, exact_count_chunk = result
            interpolated_chunks.append(interpolated_chunk)
            field_gradient_chunks.append(field_gradient_chunk)
            factor_chunks.append(factor_chunk)
            exact_count_chunks.append(exact_count_chunk)
        elif return_grad:
            interpolated_chunk, field_gradient_chunk = result
            interpolated_chunks.append(interpolated_chunk)
            field_gradient_chunks.append(field_gradient_chunk)
        else:
            interpolated_chunks.append(result)

    interpolated = torch.cat(interpolated_chunks, dim=0).reshape(B, M, C)
    if return_aux:
        return (
            interpolated,
            torch.cat(field_gradient_chunks, dim=0).reshape(B, M, D, C),
            torch.cat(factor_chunks, dim=0).reshape(B, M, D + 1, D + 1),
            torch.cat(exact_count_chunks, dim=0).reshape(B, M),
        )
    if return_grad:
        field_gradient = torch.cat(field_gradient_chunks, dim=0).reshape(B, M, D, C)
        return interpolated, field_gradient
    return interpolated


def _linear_mls_batched_head_banked_indexed_chunked_fused_forward(
    displaced_by_head: torch.Tensor,
    neighbor_positions: torch.Tensor,
    indices: torch.Tensor,
    squared_distances: torch.Tensor,
    features_by_head: torch.Tensor,
    *,
    return_grad: bool,
    chunk_size: int = _MLS_FLATTENED_CHUNK_SIZE,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    B, H, M, D = displaced_by_head.shape
    K = indices.size(-1)
    C_head = features_by_head.size(-1)

    flat_displaced_points = displaced_by_head.reshape(B * H * M, D)
    flat_neighbor_positions = neighbor_positions.reshape(B * H * M, K, D)
    flat_indices = indices.reshape(B * H * M, K)
    flat_squared_distances = squared_distances.reshape(B * H * M, K)
    feature_bank = features_by_head.permute(0, 2, 1, 3).reshape(B * H, features_by_head.size(1), C_head)

    batch_ids = torch.arange(B, device=features_by_head.device, dtype=torch.int64).view(B, 1, 1)
    head_ids = torch.arange(H, device=features_by_head.device, dtype=torch.int64).view(1, H, 1)
    feature_batch = (batch_ids * H + head_ids).expand(B, H, M).reshape(B * H * M)

    interpolated_chunks = []
    field_gradient_chunks = [] if return_grad else None

    total_queries = flat_displaced_points.size(0)
    for start in range(0, total_queries, chunk_size):
        end = min(start + chunk_size, total_queries)
        result = _linear_mls_fused_forward(
            flat_displaced_points[start:end],
            flat_neighbor_positions[start:end],
            flat_indices[start:end],
            flat_squared_distances[start:end],
            feature_bank,
            feature_batch[start:end],
            return_grad=return_grad,
        )
        if return_grad:
            interpolated_chunk, field_gradient_chunk = result
            interpolated_chunks.append(interpolated_chunk)
            field_gradient_chunks.append(field_gradient_chunk)
        else:
            interpolated_chunks.append(result)

    interpolated = torch.cat(interpolated_chunks, dim=0).reshape(B, H, M, C_head).permute(0, 2, 1, 3).contiguous()
    if return_grad:
        field_gradient = (
            torch.cat(field_gradient_chunks, dim=0)
            .reshape(B, H, M, D, C_head)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        return interpolated, field_gradient
    return interpolated


def bvh_mls_interpolate(
    points: torch.Tensor,
    displaced_points: torch.Tensor,
    features: torch.Tensor,
    k: int = 8,
    *,
    return_grad: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Interpolate features at displaced query points with local linear MLS.

    BVH construction and discrete neighbor selection are detached. Gradients flow
    through the MLS solve to ``features`` and ``displaced_points``.

    If ``return_grad=True``, returns ``(interpolated, field_gradient)``; otherwise
    returns ``interpolated`` only. ``field_gradient`` is the spatial derivative of
    the interpolated field, not a PyTorch autograd gradient.
    """
    _validate_mls_inputs(points, displaced_points, features, k)
    points = _as_contiguous(points)
    displaced_points = _as_contiguous(displaced_points)
    features = _as_contiguous(features)
    indices, squared_distances, neighbor_positions = BVHQuery.apply(points, displaced_points, k)
    result = _linear_mls_indexed_chunked_fused_forward(
        displaced_points,
        neighbor_positions,
        indices,
        squared_distances,
        features,
        return_grad=return_grad,
    )
    if return_grad:
        return result
    return result


def bvh_mls_interpolate_batched(
    points: torch.Tensor,
    displaced_points: torch.Tensor,
    features: torch.Tensor,
    k: int = 8,
    *,
    return_grad: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Batched MLS interpolation over fixed-size point-cloud batches.

    If ``return_grad=True``, returns ``(interpolated, field_gradient)``; otherwise
    returns ``interpolated`` only.
    """
    _validate_batched_mls_inputs(points, displaced_points, features, k)
    points = _as_contiguous(points)
    displaced_points = _as_contiguous(displaced_points)
    features = _as_contiguous(features)
    indices, squared_distances, neighbor_positions = BatchedBVHQuery.apply(points, displaced_points, k)
    result = _linear_mls_batched_indexed_chunked_fused_forward(
        displaced_points,
        neighbor_positions,
        indices,
        squared_distances,
        features,
        return_grad=return_grad,
    )
    if return_grad:
        return result
    return result


def _bvh_mls_interpolate_batched_head_banked(
    points: torch.Tensor,
    displaced_by_head: torch.Tensor,
    features_by_head: torch.Tensor,
    k: int = 8,
    *,
    return_grad: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Private packed-head MLS path for block benchmarks.

    ``points`` is shared across heads, while ``displaced_by_head`` and
    ``features_by_head`` carry head-specific query positions and feature banks.
    BVH construction and k-NN traversal run once per batch sample over all head
    queries.
    """
    if points.dim() != 3:
        raise ValueError("_bvh_mls_interpolate_batched_head_banked: points must have shape (B, N, D)")
    if displaced_by_head.dim() != 4:
        raise ValueError(
            "_bvh_mls_interpolate_batched_head_banked: displaced_by_head must have shape (B, H, M, D)"
        )
    if features_by_head.dim() != 4:
        raise ValueError(
            "_bvh_mls_interpolate_batched_head_banked: features_by_head must have shape (B, N, H, C_head)"
        )
    B, N, D = points.shape
    Bq, H, M, Dq = displaced_by_head.shape
    Bf, Nf, Hf, _C_head = features_by_head.shape
    _validate_supported_k("_bvh_mls_interpolate_batched_head_banked", k)
    if D not in SUPPORTED_DIMS:
        raise ValueError("_bvh_mls_interpolate_batched_head_banked: D must be 2 or 3")
    if Bq != B or Bf != B:
        raise ValueError("_bvh_mls_interpolate_batched_head_banked: batch dimensions must match")
    if Dq != D:
        raise ValueError("_bvh_mls_interpolate_batched_head_banked: displaced last dimension must match points")
    if Nf != N:
        raise ValueError("_bvh_mls_interpolate_batched_head_banked: features must match points")
    if Hf != H:
        raise ValueError("_bvh_mls_interpolate_batched_head_banked: feature heads must match displaced heads")
    if N < k:
        raise ValueError("_bvh_mls_interpolate_batched_head_banked: points must contain at least k rows per sample")
    if points.device != displaced_by_head.device or points.device != features_by_head.device:
        raise ValueError("_bvh_mls_interpolate_batched_head_banked: inputs must be on the same device")
    if points.dtype != torch.float32 or displaced_by_head.dtype != torch.float32:
        raise ValueError("_bvh_mls_interpolate_batched_head_banked: points and displaced_by_head must be float32")
    if features_by_head.dtype != torch.float32:
        raise ValueError("_bvh_mls_interpolate_batched_head_banked: features_by_head must be float32")
    if not points.is_cuda or not displaced_by_head.is_cuda or not features_by_head.is_cuda:
        raise ValueError("_bvh_mls_interpolate_batched_head_banked: inputs must be CUDA tensors")

    points = _as_contiguous(points)
    displaced_by_head = _as_contiguous(displaced_by_head)
    features_by_head = _as_contiguous(features_by_head)
    points_detached = points.detach().contiguous()
    query = displaced_by_head.detach().reshape(B, H * M, D).contiguous()

    with _temporary_bvh(build_bvh_batched, points_detached) as bvh:
        indices, squared_distances, neighbor_positions = query_knn_batched(
            bvh,
            query,
            k,
            source_points=points_detached,
            sort_queries=True,
        )

    result = _linear_mls_batched_head_banked_indexed_chunked_fused_forward(
        displaced_by_head,
        neighbor_positions,
        indices,
        squared_distances,
        features_by_head,
        return_grad=return_grad,
        chunk_size=4 * _MLS_FLATTENED_CHUNK_SIZE,
    )
    if return_grad:
        return result
    return result


def mls_interpolate(
    points: torch.Tensor,
    displaced_points: torch.Tensor,
    features: torch.Tensor,
    k: int = 8,
    *,
    return_grad: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """MLS interpolation. Dispatches on points.ndim: 2→single, 3→batched.

    If ``return_grad=True``, returns ``(interpolated, field_gradient)``; otherwise
    returns ``interpolated`` only.
    """
    if points.dim() == 3:
        return bvh_mls_interpolate_batched(points, displaced_points, features, k, return_grad=return_grad)
    return bvh_mls_interpolate(points, displaced_points, features, k, return_grad=return_grad)
