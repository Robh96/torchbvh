"""Explicit, immutable point geometry shared within a model forward."""

from contextlib import contextmanager

import torch

from ._handles import _temporary_bvh
from ._query import _build_bvh_batched


class PointGeometry:
    """Own a detached point snapshot and one batched BVH.

    Accepts float32 CUDA points of shape (N, D) or (B, N, D). Pass the same
    original tensor to MLS with ``geometry=``. Source mutation invalidates
    the object; create a new object each forward. Ordinary PyTorch version
    tracking is required (do not mutate through .data or external pointers).
    Closing the object after forward is safe: autograd retains its snapshot.
    """

    def __init__(self, points: torch.Tensor, *, experimental_fast: bool = False):
        if not isinstance(points, torch.Tensor) or points.ndim not in (2, 3):
            raise ValueError("PointGeometry: points must have shape (N, D) or (B, N, D)")
        if points.is_inference():
            raise ValueError("PointGeometry requires version-tracked source tensors")
        if not isinstance(experimental_fast, bool):
            raise TypeError("PointGeometry: experimental_fast must be bool")
        self.experimental_fast = experimental_fast
        self._source = points
        self._signature = self._source_signature(points)
        # Own storage, including for already contiguous inputs. A source change
        # after forward must not alter tensors saved by MLS backward.
        self._points = points.detach().clone(memory_format=torch.contiguous_format)
        if points.ndim == 2:
            self._points = self._points.unsqueeze(0)
        self._bvh = _build_bvh_batched(self._points)

    @staticmethod
    def _source_signature(points):
        return (points._version, points.data_ptr(), tuple(points.shape),
                tuple(points.stride()), points.device, points.dtype)

    @property
    def destroyed(self):
        return self._bvh.destroyed

    def validate(self, points):
        if self.destroyed:
            raise RuntimeError("PointGeometry has been destroyed")
        if points is not self._source:
            raise ValueError("PointGeometry belongs to a different source tensor")
        if self._source_signature(points) != self._signature:
            raise ValueError("PointGeometry source changed; rebuild geometry")
        return self

    @property
    def bvh(self):
        """Borrowed metadata for geometry operations; do not mutate/destroy it."""
        self.validate(self._source)
        return self._bvh

    def destroy(self):
        self._bvh.destroy()
        self._points = None
        self._source = None

    def __enter__(self):
        return self.validate(self._source)

    def __exit__(self, *args):
        self.destroy()


def _validate_geometry(points, geometry):
    if geometry is not None:
        if not isinstance(geometry, PointGeometry):
            raise TypeError("geometry must be a PointGeometry")
        geometry.validate(points)
    return geometry


@contextmanager
def _source_bvh(build_fn, points, geometry=None):
    if geometry is None:
        with _temporary_bvh(build_fn, points) as bvh:
            yield bvh
    else:
        yield geometry.bvh
