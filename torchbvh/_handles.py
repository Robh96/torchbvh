from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import torch


class _MappingBVHHandle(Mapping):
    """Shared mapping/liveness behavior for Python-owned BVH handles."""

    _destroyed_message = "BVH handle has been destroyed"

    def __init__(self, data: Mapping, *, marker: str | None = None):
        self._data = dict(data)
        if marker is not None:
            self._data[marker] = True
        self._destroyed = False

    @property
    def destroyed(self) -> bool:
        return self._destroyed

    def destroy(self) -> None:
        self._destroyed = True
        self._data.clear()

    def _require_live(self) -> dict:
        if self._destroyed:
            raise RuntimeError(self._destroyed_message)
        return self._data

    def __getitem__(self, key):
        return self._require_live()[key]

    def __iter__(self) -> Iterator:
        return iter(self._require_live())

    def __len__(self) -> int:
        return len(self._require_live())


class BVHHandle(_MappingBVHHandle):
    """Small Python lifecycle wrapper around the tensor-returning BVH payload."""


class BatchedBVHHandle(_MappingBVHHandle):
    """Python lifecycle wrapper around a fixed-size batched BVH payload."""

    _destroyed_message = "Batched BVH handle has been destroyed"

    def __init__(self, data: Mapping):
        super().__init__(data, marker="_batched")


class RaggedBVHHandle(_MappingBVHHandle):
    """Python lifecycle wrapper around packed ragged BVH metadata and payloads."""

    _destroyed_message = "Ragged BVH handle has been destroyed"

    def __init__(self, data: Mapping, handles: list[BVHHandle]):
        super().__init__(data, marker="_ragged")
        self._handles = list(handles)

    def destroy(self) -> None:
        if not self._destroyed:
            for handle in self._handles:
                handle.destroy()
        self._destroyed = True
        self._handles.clear()
        self._data.clear()

    def _require_handles(self) -> list[BVHHandle]:
        self._require_live()
        return self._handles


def _bvh_data(bvh, prefix: str = "query_knn") -> Mapping:
    if isinstance(bvh, (BatchedBVHHandle, RaggedBVHHandle)):
        raise TypeError(f"{prefix}: bvh must be a BVHHandle or mapping returned by build_bvh")
    if isinstance(bvh, BVHHandle):
        return bvh._require_live()
    if isinstance(bvh, Mapping):
        if bvh.get("_destroyed", False):
            raise RuntimeError("BVH handle has been destroyed")
        if bvh.get("_batched", False) or bvh.get("_ragged", False) or "batch_size" in bvh:
            raise TypeError(f"{prefix}: bvh must be a BVHHandle or mapping returned by build_bvh")
        return bvh
    raise TypeError(f"{prefix}: bvh must be a BVHHandle or mapping returned by build_bvh")


def _batched_bvh_data(bvh, prefix: str = "query_knn_batched") -> Mapping:
    if isinstance(bvh, BatchedBVHHandle):
        return bvh._require_live()
    if isinstance(bvh, (BVHHandle, RaggedBVHHandle)):
        raise TypeError(
            f"{prefix}: bvh must be a BatchedBVHHandle or mapping returned by build_bvh_batched"
        )
    if isinstance(bvh, Mapping):
        if bvh.get("_destroyed", False):
            raise RuntimeError("Batched BVH handle has been destroyed")
        if bvh.get("_ragged", False) or not (bvh.get("_batched", False) or "batch_size" in bvh):
            raise TypeError(
                f"{prefix}: bvh must be a BatchedBVHHandle or mapping returned by build_bvh_batched"
            )
        return bvh
    raise TypeError(
        f"{prefix}: bvh must be a BatchedBVHHandle or mapping returned by build_bvh_batched"
    )


def _ragged_bvh_data(bvh, prefix: str = "query_knn_ragged") -> tuple[Mapping, list[BVHHandle]]:
    if isinstance(bvh, RaggedBVHHandle):
        return bvh._require_live(), bvh._require_handles()
    if isinstance(bvh, (BVHHandle, BatchedBVHHandle)):
        raise TypeError(
            f"{prefix}: bvh must be a RaggedBVHHandle returned by build_bvh_ragged"
        )
    raise TypeError(f"{prefix}: bvh must be a RaggedBVHHandle returned by build_bvh_ragged")


@contextmanager
def _temporary_bvh(build_fn, points: torch.Tensor):
    handle = build_fn(points)
    try:
        yield handle
    finally:
        destroy_bvh(handle)


def destroy_bvh(bvh: BVHHandle | Mapping) -> None:
    """Destroy a Python BVH handle and release its tensor references."""
    if isinstance(bvh, RaggedBVHHandle):
        bvh.destroy()
        return
    if isinstance(bvh, BatchedBVHHandle):
        bvh.destroy()
        return
    if isinstance(bvh, BVHHandle):
        bvh.destroy()
        return
    if isinstance(bvh, dict):
        bvh.clear()
        bvh["_destroyed"] = True
        return
    raise TypeError("destroy_bvh: bvh must be a BVHHandle or dict returned by build_bvh")
