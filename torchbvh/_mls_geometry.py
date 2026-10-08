"""Shared geometry-owner MLS over native (B, N, H, C) feature banks."""

import torch
from torch.autograd.function import once_differentiable

from ._extension import _C
from ._constants import EXACT_DISTANCE_EPSILON, MLS_BANDWIDTH_MIN, MLS_REGULARIZATION


def eligible(dim: int, neighbors: int, channels: int, total_queries: int) -> bool:
    """A fixed, hardware-independent specialization with existing fallbacks."""
    return (dim in (2, 3) and neighbors == 4 and channels in (4, 8, 16, 32, 64)
            and 4096 <= total_queries <= (2**31 - 1) // 2)


class _GeometryMLSSpatialIndexed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, queries, sources, indices, distances, features, false_features,
                order, per_batch, per_head, return_grad, query_major):
        ctx.set_materialize_grads(False)
        values, slopes, state, hits = _C.mls_geometry_forward(
            queries, sources, indices, distances, features, false_features, order,
            per_batch, per_head, MLS_REGULARIZATION, MLS_BANDWIDTH_MIN,
            EXACT_DISTANCE_EPSILON, return_grad, query_major)
        ctx.per_batch, ctx.per_head, ctx.query_major = per_batch, per_head, query_major
        # Backward uses the saved bandwidth and hit mask; distances need not live
        # through the training step. Sources and geometry stay detached.
        ctx.save_for_backward(queries, sources, indices, features, false_features,
                              order, state, hits)
        return values, slopes

    @staticmethod
    @once_differentiable
    def backward(ctx, value_gradients, slope_gradients):
        queries, sources, indices, features, false_features, order, state, hits = ctx.saved_tensors
        if value_gradients is None:
            value_gradients = features.new_zeros((queries.size(0), features.size(-1)))
        if slope_gradients is None:
            slope_gradients = features.new_empty(0)
        feature_gradients, query_gradients = _C.mls_geometry_backward(
            queries, sources, indices, features.new_empty(0), features, false_features,
            order, state, hits, value_gradients.contiguous(), slope_gradients.contiguous(),
            ctx.per_batch, ctx.per_head, MLS_BANDWIDTH_MIN, EXACT_DISTANCE_EPSILON,
            ctx.query_major, ctx.needs_input_grad[0],
            ctx.needs_input_grad[4] or ctx.needs_input_grad[5])
        split = features.size(1)
        return (query_gradients if ctx.needs_input_grad[0] else None,
                None, None, None,
                feature_gradients[:, :split] if ctx.needs_input_grad[4] else None,
                feature_gradients[:, split:] if ctx.needs_input_grad[5] else None,
                None, None, None, None, None)


def interpolate(queries, sources, indices, distances, features, false_features, order,
                *, queries_per_batch, queries_per_head, return_grad,
                query_major_output=True):
    """Internal entry point shared by single-bank and routed public APIs."""
    if false_features is None:
        false_features = features.new_empty((features.size(0), 0, features.size(2), features.size(3)))
    values, slopes = _GeometryMLSSpatialIndexed.apply(
        queries, sources, indices, distances, features, false_features, order,
        queries_per_batch, queries_per_head, return_grad, query_major_output)
    return (values, slopes) if return_grad else values
