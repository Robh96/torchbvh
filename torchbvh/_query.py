from collections.abc import Mapping

import torch

from . import _C
from ._constants import SUPPORTED_DIMS
from ._handles import (
    BVHHandle,
    BatchedBVHHandle,
    RaggedBVHHandle,
    _batched_bvh_data,
    _bvh_data,
    _ragged_bvh_data,
)
from ._validation import (
    _as_contiguous,
    _as_contiguous_int64,
    _validate_cuda_float32,
    _validate_offsets,
    _validate_ragged_points,
    _validate_supported_k,
)


# ---------------------------------------------------------------------------
# Private per-variant builders
# ---------------------------------------------------------------------------

_INT32_MAX = 2**31 - 1


def _validate_narrow_build_extent(prefix: str, points: torch.Tensor) -> None:
    # The production Morton path uses 32-bit CUB values and segment offsets.
    # Fail explicitly instead of allowing a silent integer wrap in native code.
    total = int(points.size(0)) if points.dim() == 2 else int(points.size(0)) * int(points.size(1))
    if total > _INT32_MAX:
        raise ValueError(f"{prefix}: total point count must fit in signed int32")

def _build_bvh_single(points: torch.Tensor) -> BVHHandle:
    if points.dim() != 2:
        raise ValueError("build_bvh: points must have shape (N, D)")
    if points.size(1) not in SUPPORTED_DIMS:
        raise ValueError("build_bvh: D must be 2 or 3")
    _validate_cuda_float32("build_bvh", points, "points")
    points = _as_contiguous(points)
    _validate_narrow_build_extent("build_bvh", points)
    data = dict(_C.build_bvh_batched_cooperative(points.unsqueeze(0)))
    # The native cooperative implementation shares one fixed-size topology
    # across a batch. Strip that leading batch dimension to preserve the
    # established single-handle payload exactly.
    data.pop("batch_size")
    for key in ("node_aabbs", "sorted_indices", "scene_min", "scene_max"):
        data[key] = data[key].squeeze(0)
    return BVHHandle(data)


def _build_bvh_batched(points: torch.Tensor) -> BatchedBVHHandle:
    if points.dim() != 3:
        raise ValueError("build_bvh: batched points must have shape (B, N, D)")
    if points.size(2) not in SUPPORTED_DIMS:
        raise ValueError("build_bvh: D must be 2 or 3")
    _validate_cuda_float32("build_bvh", points, "points")
    points = _as_contiguous(points)
    _validate_narrow_build_extent("build_bvh", points)
    return BatchedBVHHandle(_C.build_bvh_batched_cooperative(points))


def _build_bvh_ragged(points: torch.Tensor, batch_offsets: torch.Tensor) -> RaggedBVHHandle:
    points = _as_contiguous(points)
    batch_offsets = _as_contiguous_int64(batch_offsets)
    offsets = _validate_ragged_points("build_bvh", points, batch_offsets)
    handles = [
        _build_bvh_single(points[offsets[batch] : offsets[batch + 1]].contiguous())
        for batch in range(len(offsets) - 1)
    ]
    counts = torch.diff(batch_offsets).clone()
    return RaggedBVHHandle(
        {
            "batch_offsets": batch_offsets.clone(),
            "num_leaves_per_sample": counts,
            "batch_size": len(offsets) - 1,
            "dim": int(points.size(1)),
        },
        handles,
    )


# ---------------------------------------------------------------------------
# Public unified builder
# ---------------------------------------------------------------------------

def build_bvh(
    points: torch.Tensor,
    *,
    batch_offsets: torch.Tensor | None = None,
) -> BVHHandle | BatchedBVHHandle | RaggedBVHHandle:
    """Build a BVH. Dispatches on points shape and batch_offsets.

    - ``(N, D)`` → single BVH, returns ``BVHHandle``
    - ``(B, N, D)`` → fixed-size batched BVHs, returns ``BatchedBVHHandle``
    - ``(total_N, D)`` + ``batch_offsets=(B+1,)`` → ragged packed BVHs, returns ``RaggedBVHHandle``

    Ragged neighbor indices are local to each sample.
    """
    if batch_offsets is not None:
        if points.dim() != 2:
            raise ValueError("build_bvh: ragged mode requires 2-D points (total_N, D)")
        return _build_bvh_ragged(points, batch_offsets)
    if points.dim() == 3:
        return _build_bvh_batched(points)
    if points.dim() == 2:
        return _build_bvh_single(points)
    raise ValueError("build_bvh: points must have shape (N, D) or (B, N, D)")


# ---------------------------------------------------------------------------
# Private per-variant query functions
# ---------------------------------------------------------------------------

def _query_knn_single(
    bvh: BVHHandle,
    query_points: torch.Tensor,
    k: int,
    *,
    source_points: torch.Tensor | None = None,
    sort_queries: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_supported_k("query_knn", k)
    data = _bvh_data(bvh, "query_knn")
    if query_points.dim() != 2:
        raise ValueError("query_knn: query_points must have shape (M, D)")
    if query_points.size(1) != data["dim"]:
        raise ValueError("query_knn: query_points second dimension must match BVH dim")
    _validate_cuda_float32("query_knn", query_points, "query_points")
    if query_points.device != data["sorted_indices"].device:
        raise ValueError("query_knn: query_points must be on the same device as the BVH")
    query_points = _as_contiguous(query_points)
    batched_queries = query_points.unsqueeze(0)
    node_aabbs = data["node_aabbs"].unsqueeze(0)
    sorted_indices = data["sorted_indices"].unsqueeze(0)
    if sort_queries:
        sort_perm = _C.morton_sort_queries_narrow(
            batched_queries,
            data["scene_min"].unsqueeze(0),
            data["scene_max"].unsqueeze(0),
        )
        indices, squared_distances = _C.query_knn_batched_cached_bounds_ordered(
            node_aabbs,
            sorted_indices,
            batched_queries,
            sort_perm.contiguous(),
            data["num_leaves"],
            data["num_real_nodes"],
            data["dim"],
            k,
        )
    else:
        indices, squared_distances = _C.query_knn_batched_cached_bounds(
            node_aabbs,
            sorted_indices,
            batched_queries,
            data["num_leaves"],
            data["num_real_nodes"],
            data["dim"],
            k,
        )
    indices = indices.squeeze(0)
    squared_distances = squared_distances.squeeze(0)
    if source_points is None:
        return indices, squared_distances

    _validate_single_source_points("query_knn", data, source_points)
    source_points = _as_contiguous(source_points)
    return indices, squared_distances, source_points[indices]


def _validate_single_source_points(prefix: str, data: Mapping, source_points: torch.Tensor) -> None:
    if source_points.dim() != 2:
        raise ValueError(f"{prefix}: source_points must have shape (N, D)")
    if source_points.size(0) != data["num_leaves"]:
        raise ValueError(f"{prefix}: source_points first dimension must match BVH num_leaves")
    if source_points.size(1) != data["dim"]:
        raise ValueError(f"{prefix}: source_points second dimension must match BVH dim")
    _validate_cuda_float32(prefix, source_points, "source_points")
    if source_points.device != data["sorted_indices"].device:
        raise ValueError(f"{prefix}: source_points must be on the same device as the BVH")


def _validate_batched_source_points(prefix: str, data: Mapping, source_points: torch.Tensor) -> None:
    if source_points.dim() != 3:
        raise ValueError(f"{prefix}: source_points must have shape (B, N, D)")
    if source_points.size(0) != data["batch_size"]:
        raise ValueError(f"{prefix}: source_points batch size must match BVH batch_size")
    if source_points.size(1) != data["num_leaves"]:
        raise ValueError(f"{prefix}: source_points second dimension must match BVH num_leaves")
    if source_points.size(2) != data["dim"]:
        raise ValueError(f"{prefix}: source_points last dimension must match BVH dim")
    _validate_cuda_float32(prefix, source_points, "source_points")
    if source_points.device != data["sorted_indices"].device:
        raise ValueError(f"{prefix}: source_points must be on the same device as the BVH")


def _gather_batched_neighbor_positions(
    source_points: torch.Tensor,
    indices: torch.Tensor,
    query_count: int,
    dim: int,
) -> torch.Tensor:
    batch_size, _, neighbor_count = indices.shape
    gather_index = indices.reshape(
        batch_size, query_count * neighbor_count, 1
    ).expand(-1, -1, dim)
    return torch.gather(source_points, 1, gather_index).reshape(
        batch_size, query_count, neighbor_count, dim
    )


def _query_knn_batched(
    bvh: BatchedBVHHandle,
    query_points: torch.Tensor,
    k: int,
    *,
    source_points: torch.Tensor | None = None,
    sort_queries: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_supported_k("query_knn", k)
    data = _batched_bvh_data(bvh, "query_knn")
    if query_points.dim() != 3:
        raise ValueError("query_knn: batched query_points must have shape (B, M, D)")
    if query_points.size(0) != data["batch_size"]:
        raise ValueError("query_knn: query_points batch size must match BVH batch_size")
    if query_points.size(2) != data["dim"]:
        raise ValueError("query_knn: query_points last dimension must match BVH dim")
    _validate_cuda_float32("query_knn", query_points, "query_points")
    if query_points.device != data["sorted_indices"].device:
        raise ValueError("query_knn: query_points must be on the same device as the BVH")
    query_points = _as_contiguous(query_points)

    if sort_queries:
        query_points = query_points.contiguous()
        sort_perm = _C.morton_sort_queries_narrow(
            query_points,
            data["scene_min"],
            data["scene_max"],
        )
        indices, squared_distances = _C.query_knn_batched_cached_bounds_ordered(
            data["node_aabbs"],
            data["sorted_indices"],
            query_points,
            sort_perm.contiguous(),
            data["num_leaves"],
            data["num_real_nodes"],
            data["dim"],
            k,
        )
    else:
        indices, squared_distances = _C.query_knn_batched_cached_bounds(
            data["node_aabbs"],
            data["sorted_indices"],
            query_points,
            data["num_leaves"],
            data["num_real_nodes"],
            data["dim"],
            k,
        )
    if source_points is None:
        return indices, squared_distances

    _validate_batched_source_points("query_knn", data, source_points)
    source_points = _as_contiguous(source_points)
    neighbor_positions = _gather_batched_neighbor_positions(
        source_points,
        indices,
        query_points.size(1),
        int(data["dim"]),
    )
    return indices, squared_distances, neighbor_positions


def _validate_ragged_source_points(
    prefix: str,
    data: Mapping,
    source_points: torch.Tensor,
    *,
    total_rows: int,
    device: torch.device,
) -> None:
    if source_points.dim() != 2:
        raise ValueError(f"{prefix}: ragged source_points must have shape (total_N, D)")
    if source_points.size(0) != total_rows:
        raise ValueError(f"{prefix}: source_points first dimension must match BVH total_N")
    if source_points.size(1) != data["dim"]:
        raise ValueError(f"{prefix}: source_points second dimension must match BVH dim")
    _validate_cuda_float32(prefix, source_points, "source_points")
    if source_points.device != device:
        raise ValueError(f"{prefix}: source_points must be on the same device as the BVH")


def _query_knn_ragged(
    bvh: RaggedBVHHandle,
    query_points: torch.Tensor,
    query_offsets: torch.Tensor,
    k: int,
    *,
    source_points: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_supported_k("query_knn", k)

    data, handles = _ragged_bvh_data(bvh, "query_knn")
    if query_points.dim() != 2:
        raise ValueError("query_knn: ragged query_points must have shape (total_M, D)")
    if query_points.size(1) != data["dim"]:
        raise ValueError("query_knn: query_points second dimension must match BVH dim")
    _validate_cuda_float32("query_knn", query_points, "query_points")
    if query_points.device != data["batch_offsets"].device:
        raise ValueError("query_knn: query_points must be on the same device as the BVH")
    query_points = _as_contiguous(query_points)
    query_offsets = _as_contiguous_int64(query_offsets)

    q_offsets = _validate_offsets(
        "query_knn",
        query_offsets,
        total_rows=int(query_points.size(0)),
        device=query_points.device,
    )
    if len(q_offsets) - 1 != data["batch_size"]:
        raise ValueError("query_knn: query_offsets batch size must match BVH batch_size")

    batch_offsets = [int(v) for v in data["batch_offsets"].detach().cpu().tolist()]
    counts = [right - left for left, right in zip(batch_offsets, batch_offsets[1:])]
    if any(count < k for count in counts):
        raise ValueError("query_knn: each source sample must contain at least k points")

    if source_points is not None:
        _validate_ragged_source_points(
            "query_knn",
            data,
            source_points,
            total_rows=batch_offsets[-1],
            device=query_points.device,
        )
        source_points = _as_contiguous(source_points)

    all_indices = []
    all_distances = []
    all_positions = []
    for batch, handle in enumerate(handles):
        q_start, q_end = q_offsets[batch], q_offsets[batch + 1]
        sample_queries = query_points[q_start:q_end].contiguous()
        if source_points is None:
            indices, distances = _query_knn_single(handle, sample_queries, k)
        else:
            p_start, p_end = batch_offsets[batch], batch_offsets[batch + 1]
            sample_points = source_points[p_start:p_end].contiguous()
            indices, distances, positions = _query_knn_single(
                handle, sample_queries, k, source_points=sample_points
            )
            all_positions.append(positions)
        all_indices.append(indices)
        all_distances.append(distances)

    packed_indices = torch.cat(all_indices, dim=0)
    packed_distances = torch.cat(all_distances, dim=0)
    if source_points is None:
        return packed_indices, packed_distances
    return packed_indices, packed_distances, torch.cat(all_positions, dim=0)


# ---------------------------------------------------------------------------
# Public unified query
# ---------------------------------------------------------------------------

def query_knn(
    bvh: BVHHandle | BatchedBVHHandle | RaggedBVHHandle,
    query_points: torch.Tensor,
    k: int,
    *,
    query_offsets: torch.Tensor | None = None,
    source_points: torch.Tensor | None = None,
    sort_queries: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Query exact k-NN. Dispatches on handle type.

    - ``BVHHandle`` → single-sample, returns shape ``(M, k)``
    - ``BatchedBVHHandle`` → batched, returns shape ``(B, M, k)``
    - ``RaggedBVHHandle`` → packed ragged; ``query_offsets`` required, returns packed ``(total_M, k)``

    Returns ``(indices, squared_distances)`` or, when ``source_points`` is provided,
    ``(indices, squared_distances, neighbor_positions)``.
    """
    if isinstance(bvh, RaggedBVHHandle):
        if query_offsets is None:
            raise ValueError(
                "query_knn: query_offsets is required when querying a ragged BVH"
            )
        return _query_knn_ragged(bvh, query_points, query_offsets, k, source_points=source_points)
    if query_offsets is not None:
        raise TypeError("query_knn: query_offsets requires a RaggedBVHHandle")
    if isinstance(bvh, BatchedBVHHandle):
        return _query_knn_batched(
            bvh, query_points, k, source_points=source_points, sort_queries=sort_queries
        )
    return _query_knn_single(
        bvh, query_points, k, source_points=source_points, sort_queries=sort_queries
    )
