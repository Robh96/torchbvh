import torch

from ._handles import BVHHandle, BatchedBVHHandle, RaggedBVHHandle, destroy_bvh
from ._mls import mls_interpolate
from ._query import (
    _build_bvh_batched,
    _build_bvh_ragged,
    _build_bvh_single,
    query_knn,
)
from ._validation import _as_contiguous, _as_contiguous_int64


class BVH:
    """Unified BVH wrapper. Dispatches on points shape and batch_offsets.

    - ``BVH(points)`` where ``points`` is ``(N, D)`` → single-sample BVH
    - ``BVH(points)`` where ``points`` is ``(B, N, D)`` → fixed-size batched BVH
    - ``BVH(points, batch_offsets=offsets)`` where ``points`` is ``(total_N, D)``
      → ragged packed BVH; ``interpolate()`` is not supported on ragged instances.

    Owns handle lifetime. Supports context-manager protocol.
    Does not expose internal handle fields.
    """

    def __init__(
        self,
        points: torch.Tensor,
        *,
        batch_offsets: torch.Tensor | None = None,
    ) -> None:
        if batch_offsets is not None:
            if points.dim() != 2:
                raise ValueError("BVH: ragged mode requires 2-D points (total_N, D)")
            points = _as_contiguous(points)
            batch_offsets = _as_contiguous_int64(batch_offsets)
            self._handle: BVHHandle | BatchedBVHHandle | RaggedBVHHandle = _build_bvh_ragged(
                points, batch_offsets
            )
        elif points.dim() == 3:
            points = _as_contiguous(points)
            self._handle = _build_bvh_batched(points)
        elif points.dim() == 2:
            points = _as_contiguous(points)
            self._handle = _build_bvh_single(points)
        else:
            raise ValueError("BVH: points must have shape (N, D) or (B, N, D)")
        self._points = points

    @property
    def destroyed(self) -> bool:
        return self._handle.destroyed

    def knn(
        self,
        query_points: torch.Tensor,
        k: int,
        *,
        query_offsets: torch.Tensor | None = None,
        source_points: torch.Tensor | None = None,
        sort_queries: bool = True,
    ):
        """Query k nearest neighbors.

        For ragged BVHs ``query_offsets`` is required. For single/batched BVHs
        ``sort_queries`` controls Morton-sorted traversal (default True).
        """
        return query_knn(
            self._handle,
            query_points,
            k,
            query_offsets=query_offsets,
            source_points=source_points,
            sort_queries=sort_queries,
        )

    def interpolate(
        self,
        displaced_points: torch.Tensor,
        features: torch.Tensor,
        k: int = 8,
        *,
        return_grad: bool = False,
    ):
        """MLS interpolation at displaced query positions.

        Not supported for ragged BVH instances.
        """
        if isinstance(self._handle, RaggedBVHHandle):
            raise TypeError(
                "BVH.interpolate() is not supported for ragged inputs; "
                "use a single-sample or batched BVH"
            )
        return mls_interpolate(
            self._points, displaced_points, features, k, return_grad=return_grad
        )

    def destroy(self) -> None:
        destroy_bvh(self._handle)

    def __enter__(self) -> "BVH":
        return self

    def __exit__(self, *args) -> None:
        self.destroy()


class BatchedBVH(BVH):
    """Convenience subclass for fixed-size batched BVH. Equivalent to BVH(points) with (B, N, D) input."""

    def __init__(self, points: torch.Tensor) -> None:
        if points.dim() != 3:
            raise ValueError("BatchedBVH: points must have shape (B, N, D)")
        super().__init__(points)


class RaggedBVH(BVH):
    """Convenience subclass for ragged (variable-size) batched BVH.

    Equivalent to BVH(points, batch_offsets=batch_offsets).
    ``interpolate()`` raises ``TypeError``.
    """

    def __init__(self, points: torch.Tensor, batch_offsets: torch.Tensor) -> None:
        super().__init__(points, batch_offsets=batch_offsets)
