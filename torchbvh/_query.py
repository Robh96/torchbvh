from collections.abc import Mapping

import torch

from . import _C
from ._handles import (
    BVHHandle,
    BatchedBVHHandle,
    RaggedBVHHandle,
    _batched_bvh_data,
    _bvh_data,
    _ragged_bvh_data,
)
from ._validation import (
    _validate_cuda_float32_contiguous,
    _validate_offsets,
    _validate_ragged_points,
    _validate_supported_k,
)


# ---------------------------------------------------------------------------
# Private per-variant builders
# ---------------------------------------------------------------------------

def _build_bvh_single(points: torch.Tensor) -> BVHHandle:
    return BVHHandle(_C.build_bvh(points))


def _build_bvh_batched(points: torch.Tensor) -> BatchedBVHHandle:
    return BatchedBVHHandle(_C.build_bvh_batched(points))


def _build_bvh_ragged(points: torch.Tensor, batch_offsets: torch.Tensor) -> RaggedBVHHandle:
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


# Backward-compatible per-variant aliases.
build_bvh_batched = _build_bvh_batched
build_bvh_ragged = _build_bvh_ragged


# ---------------------------------------------------------------------------
# Private per-variant query functions
# ---------------------------------------------------------------------------

def _query_knn_single(
    bvh: BVHHandle | Mapping,
    query_points: torch.Tensor,
    k: int,
    *,
    source_points: torch.Tensor | None = None,
    sort_queries: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_supported_k("query_knn", k)
    data = _bvh_data(bvh, "query_knn")
    if sort_queries:
        from ._reorder import morton_sort_queries_batched

        query_points = query_points.contiguous()
        query_batch = query_points.unsqueeze(0)
        sort_perm, _ = morton_sort_queries_batched(
            query_batch,
            data["scene_min"].unsqueeze(0),
            data["scene_max"].unsqueeze(0),
        )
        indices, squared_distances = _C.query_knn_ordered(
            data["node_aabbs"],
            data["sorted_indices"],
            query_points,
            sort_perm.squeeze(0).contiguous(),
            data["num_leaves"],
            data["leaf_level"],
            data["dim"],
            k,
        )
    else:
        indices, squared_distances = _C.query_knn(
            data["node_aabbs"],
            data["sorted_indices"],
            query_points,
            data["num_leaves"],
            data["leaf_level"],
            data["dim"],
            k,
        )
    if source_points is None:
        return indices, squared_distances

    _validate_single_source_points("query_knn", data, source_points)
    return indices, squared_distances, source_points[indices]


def _validate_single_source_points(prefix: str, data: Mapping, source_points: torch.Tensor) -> None:
    if source_points.dim() != 2:
        raise ValueError(f"{prefix}: source_points must have shape (N, D)")
    if source_points.size(0) != data["num_leaves"]:
        raise ValueError(f"{prefix}: source_points first dimension must match BVH num_leaves")
    if source_points.size(1) != data["dim"]:
        raise ValueError(f"{prefix}: source_points second dimension must match BVH dim")


def _validate_batched_source_points(prefix: str, data: Mapping, source_points: torch.Tensor) -> None:
    if source_points.dim() != 3:
        raise ValueError(f"{prefix}: source_points must have shape (B, N, D)")
    if source_points.size(0) != data["batch_size"]:
        raise ValueError(f"{prefix}: source_points batch size must match BVH batch_size")
    if source_points.size(1) != data["num_leaves"]:
        raise ValueError(f"{prefix}: source_points second dimension must match BVH num_leaves")
    if source_points.size(2) != data["dim"]:
        raise ValueError(f"{prefix}: source_points last dimension must match BVH dim")
    _validate_cuda_float32_contiguous(prefix, source_points, "source_points")
    if source_points.device != data["sorted_indices"].device:
        raise ValueError(f"{prefix}: source_points must be on the same device as the BVH")


def _gather_batched_neighbor_positions(
    source_points: torch.Tensor,
    indices: torch.Tensor,
    query_count: int,
    dim: int,
) -> torch.Tensor:
    gather_index = indices.unsqueeze(-1).expand(-1, -1, -1, dim)
    expanded_source = source_points.unsqueeze(1).expand(-1, query_count, -1, -1)
    return torch.gather(expanded_source, 2, gather_index)


def _query_knn_batched(
    bvh: BatchedBVHHandle | Mapping,
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
    _validate_cuda_float32_contiguous("query_knn", query_points, "query_points")
    if query_points.device != data["sorted_indices"].device:
        raise ValueError("query_knn: query_points must be on the same device as the BVH")

    if sort_queries:
        from ._reorder import morton_sort_queries_batched

        query_points = query_points.contiguous()
        sort_perm, _ = morton_sort_queries_batched(
            query_points,
            data["scene_min"],
            data["scene_max"],
        )
        indices, squared_distances = _C.query_knn_batched_ordered(
            data["node_aabbs"],
            data["sorted_indices"],
            query_points,
            sort_perm.contiguous(),
            data["num_leaves"],
            data["num_real_nodes"],
            data["leaf_level"],
            data["dim"],
            k,
        )
    else:
        indices, squared_distances = _C.query_knn_batched(
            data["node_aabbs"],
            data["sorted_indices"],
            query_points,
            data["num_leaves"],
            data["num_real_nodes"],
            data["leaf_level"],
            data["dim"],
            k,
        )
    if source_points is None:
        return indices, squared_distances

    _validate_batched_source_points("query_knn", data, source_points)
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
    _validate_cuda_float32_contiguous(prefix, source_points, "source_points")
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
    _validate_cuda_float32_contiguous("query_knn", query_points, "query_points")
    if query_points.device != data["batch_offsets"].device:
        raise ValueError("query_knn: query_points must be on the same device as the BVH")

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
    bvh: BVHHandle | BatchedBVHHandle | RaggedBVHHandle | Mapping,
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
    if isinstance(bvh, BatchedBVHHandle):
        return _query_knn_batched(
            bvh, query_points, k, source_points=source_points, sort_queries=sort_queries
        )
    if isinstance(bvh, Mapping) and bvh.get("_batched", False):
        return _query_knn_batched(
            bvh, query_points, k, source_points=source_points, sort_queries=sort_queries
        )
    return _query_knn_single(
        bvh, query_points, k, source_points=source_points, sort_queries=sort_queries
    )


# Backward-compatible per-variant aliases.
query_knn_batched = _query_knn_batched
query_knn_ragged = _query_knn_ragged
