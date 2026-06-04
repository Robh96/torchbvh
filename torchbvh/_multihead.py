"""Stable displaced-query and matching-head interpolation helpers."""

import torch

from ._constants import EXACT_DISTANCE_EPSILON, SUPPORTED_DIMS
from ._handles import _temporary_bvh
from ._query import build_bvh_batched, query_knn_batched
from ._validation import _validate_cuda_float32_contiguous

__all__ = [
    "query_displaced_knn",
    "gather_neighbor_values",
    "interpolate_displaced",
]


def _validate_displaced_inputs(pos: torch.Tensor, q: torch.Tensor, prefix: str) -> None:
    if pos.dim() != 3:
        raise ValueError(f"{prefix}: pos must have shape (B, N, D)")
    if q.dim() != 4:
        raise ValueError(f"{prefix}: q must have shape (B, N, H, D)")
    if pos.size(2) not in SUPPORTED_DIMS:
        raise ValueError(f"{prefix}: D must be 2 or 3")
    if q.size(0) != pos.size(0):
        raise ValueError(f"{prefix}: q batch size must match pos")
    if q.size(1) != pos.size(1):
        raise ValueError(f"{prefix}: q point dimension must match pos")
    if q.size(3) != pos.size(2):
        raise ValueError(f"{prefix}: q last dimension must match pos")
    if q.size(2) < 1:
        raise ValueError(f"{prefix}: q must contain at least one head")
    _validate_cuda_float32_contiguous(prefix, pos, "pos")
    _validate_cuda_float32_contiguous(prefix, q, "q")
    if pos.device != q.device:
        raise ValueError(f"{prefix}: pos and q must be on the same device")


def query_displaced_knn(
    pos: torch.Tensor,
    q: torch.Tensor,
    k: int = 8,
    *,
    return_positions: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Query k-NN for per-point, per-head displaced positions.

    Builds one BVH per sample over ``pos: (B, N, D)`` and flattens only the
    queries ``q: (B, N, H, D)`` to ``(B, N * H, D)``. Returned neighbor indices
    are local to each sample. The flattened queries use the default
    Morton-sorted traversal path for high-volume query locality.
    """
    _validate_displaced_inputs(pos, q, "query_displaced_knn")
    batch_size, n, heads, dim = q.shape
    flat_queries = q.reshape(batch_size, n * heads, dim).contiguous()
    source_pos = pos.detach().contiguous()
    with _temporary_bvh(build_bvh_batched, source_pos) as batched_bvh:
        if return_positions:
            indices, squared_distances, neighbor_positions = query_knn_batched(
                batched_bvh,
                flat_queries,
                k,
                source_points=source_pos,
                sort_queries=True,
            )
            return (
                indices.reshape(batch_size, n, heads, k).contiguous(),
                squared_distances.reshape(batch_size, n, heads, k).contiguous(),
                neighbor_positions.reshape(batch_size, n, heads, k, dim).contiguous(),
            )

        indices, squared_distances = query_knn_batched(
            batched_bvh,
            flat_queries,
            k,
            sort_queries=True,
        )
        return (
            indices.reshape(batch_size, n, heads, k).contiguous(),
            squared_distances.reshape(batch_size, n, heads, k).contiguous(),
        )


def gather_neighbor_values(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather per-head source values for displaced-query neighborhoods.

    ``values`` has shape ``(B, N, H, Ch)`` and ``indices`` has shape
    ``(B, N, H, k)``. The output has shape ``(B, N, H, k, Ch)`` and gathers
    from the matching head only.
    """
    if values.dim() != 4:
        raise ValueError("gather_neighbor_values: values must have shape (B, N, H, Ch)")
    if indices.dim() != 4:
        raise ValueError("gather_neighbor_values: indices must have shape (B, N, H, k)")
    if values.size(0) != indices.size(0):
        raise ValueError("gather_neighbor_values: indices batch size must match values")
    if values.size(1) != indices.size(1):
        raise ValueError("gather_neighbor_values: indices point dimension must match values")
    if values.size(2) != indices.size(2):
        raise ValueError("gather_neighbor_values: indices head dimension must match values")
    if values.size(3) < 1:
        raise ValueError("gather_neighbor_values: values must have at least one channel")
    _validate_cuda_float32_contiguous("gather_neighbor_values", values, "values")
    if indices.dtype != torch.int64:
        raise ValueError("gather_neighbor_values: indices must be int64")
    if values.device != indices.device:
        raise ValueError("gather_neighbor_values: values and indices must be on the same device")
    if not indices.is_contiguous():
        raise ValueError("gather_neighbor_values: indices must be contiguous")
    if indices.numel() > 0:
        amin, amax = torch.aminmax(indices)
        if int(amin) < 0 or int(amax) >= values.size(1):
            raise ValueError("gather_neighbor_values: indices must be local source indices")

    k = indices.size(-1)
    channels = values.size(-1)
    expanded_values = values.unsqueeze(3).expand(-1, -1, -1, k, -1)
    gather_index = indices.unsqueeze(-1).expand(-1, -1, -1, -1, channels)
    return torch.gather(expanded_values, 1, gather_index)


def _weighted_mean_rank_loop(
    values: torch.Tensor,
    indices: torch.Tensor,
    exact_mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    batch_size, n, heads, channels = values.shape
    weighted = values.new_zeros((batch_size, n, heads, channels))
    exact_sum = values.new_zeros((batch_size, n, heads, channels))
    batch_index = torch.arange(batch_size, device=values.device).view(batch_size, 1, 1)
    head_index = torch.arange(heads, device=values.device).view(1, 1, heads)

    for neighbor_rank in range(indices.size(-1)):
        neighbor_values = values[batch_index, indices[..., neighbor_rank], head_index, :]
        weighted = weighted + neighbor_values * weights[..., neighbor_rank].unsqueeze(-1)
        exact_sum = exact_sum + neighbor_values * exact_mask[..., neighbor_rank].to(values.dtype).unsqueeze(-1)

    if exact_mask.any():
        exact_counts = exact_mask.to(values.dtype).sum(dim=-1, keepdim=True).clamp(min=1.0)
        exact_average = exact_sum / exact_counts
        weighted = torch.where(exact_mask.any(dim=-1, keepdim=True), exact_average, weighted)
    return weighted


def interpolate_displaced(
    pos: torch.Tensor,
    q: torch.Tensor,
    values: torch.Tensor,
    k: int = 8,
    *,
    reduction: str = "weighted_mean",
) -> torch.Tensor:
    """Interpolate matching-head values at displaced query positions.

    ``weighted_mean`` uses inverse squared-distance weights
    ``w = 1 / (squared_distance + 1e-12)`` normalized over the ``k`` neighbors.
    If a query has exact zero-distance neighbors, the output is the unweighted
    mean of only those exact-hit neighbor values.
    """
    if reduction != "weighted_mean":
        raise ValueError("interpolate_displaced: reduction must be 'weighted_mean'")
    _validate_displaced_inputs(pos, q, "interpolate_displaced")
    if values.dim() != 4:
        raise ValueError("interpolate_displaced: values must have shape (B, N, H, Ch)")
    if values.size(0) != pos.size(0) or values.size(1) != pos.size(1):
        raise ValueError("interpolate_displaced: values first two dimensions must match pos")
    if values.size(2) != q.size(2):
        raise ValueError("interpolate_displaced: values head dimension must match q")
    if values.size(3) < 1:
        raise ValueError("interpolate_displaced: values must have at least one channel")
    _validate_cuda_float32_contiguous("interpolate_displaced", values, "values")
    if values.device != pos.device:
        raise ValueError("interpolate_displaced: values must be on the same device as pos")

    indices, squared_distances = query_displaced_knn(
        pos,
        q,
        k,
        return_positions=False,
    )
    exact_mask = squared_distances <= EXACT_DISTANCE_EPSILON
    weights = torch.reciprocal(squared_distances + EXACT_DISTANCE_EPSILON)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=EXACT_DISTANCE_EPSILON)

    return _weighted_mean_rank_loop(values, indices, exact_mask, weights)
