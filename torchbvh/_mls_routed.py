"""MLS with direct native-layout feature banks; shares the existing solve."""
import torch
from torch.autograd.function import once_differentiable

from ._extension import _C
from ._constants import EXACT_DISTANCE_EPSILON, MLS_BANDWIDTH_MIN, MLS_REGULARIZATION

_CHANNEL_TILE = 1
_SHARED_COEFFICIENTS = False
_COEFFICIENT_SAFE_RATIO = .05
_QUERY_MAJOR_OUTPUT = False
_AGGREGATE_SCATTER = False


class _RoutedMLSSpatialIndexed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, queries, sources, indices, distances, true_features,
                false_features, order, per_batch, per_head, return_grad, channel_tile, shared_coefficients, safe_ratio, query_major_output, aggregate_scatter):
        ctx.set_materialize_grads(False)
        values, slopes, factors, exact_counts = _C.mls_routed_indexed_forward(
            queries, sources, indices, distances, true_features, false_features,
            order, per_batch, per_head, MLS_REGULARIZATION, MLS_BANDWIDTH_MIN,
            EXACT_DISTANCE_EPSILON, return_grad, channel_tile, shared_coefficients, safe_ratio, query_major_output)
        ctx.per_batch, ctx.per_head = per_batch, per_head
        ctx.channel_tile = channel_tile
        ctx.query_major_output = query_major_output
        ctx.aggregate_scatter = aggregate_scatter
        ctx.save_for_backward(queries, sources, indices, distances, true_features,
                              false_features, order, factors, exact_counts)
        return values, slopes

    @staticmethod
    @once_differentiable
    def backward(ctx, d_values, d_slopes):
        queries, sources, indices, distances, true_features, false_features, order, factors, counts = ctx.saved_tensors
        if d_values is None:
            d_values = torch.zeros((queries.size(0), true_features.size(-1)),
                                   device=queries.device, dtype=queries.dtype)
        if d_slopes is None:
            d_slopes = torch.empty(0, device=queries.device, dtype=queries.dtype)
        d_features, d_queries = _C.mls_routed_indexed_backward(
            queries, sources, indices, distances, true_features, false_features,
            order, factors, counts, d_values.contiguous(), d_slopes.contiguous(),
            ctx.per_batch, ctx.per_head, MLS_BANDWIDTH_MIN, EXACT_DISTANCE_EPSILON, ctx.channel_tile,
            ctx.query_major_output, ctx.aggregate_scatter, ctx.needs_input_grad[0],
            ctx.needs_input_grad[4] or ctx.needs_input_grad[5])
        split = true_features.size(1)
        return (d_queries if ctx.needs_input_grad[0] else None, None, None, None,
                d_features[:, :split] if ctx.needs_input_grad[4] else None,
                d_features[:, split:] if ctx.needs_input_grad[5] else None,
                None, None, None, None, None, None, None, None, None)


def _routed_mls_spatial_indexed(queries, sources, indices, distances, true_features,
                              false_features, order, *, queries_per_batch,
                              queries_per_head, return_grad, query_major_output=None):
    values, slopes = _RoutedMLSSpatialIndexed.apply(
        queries, sources, indices, distances, true_features, false_features,
        order, queries_per_batch, queries_per_head, return_grad, _CHANNEL_TILE,
        _SHARED_COEFFICIENTS, _COEFFICIENT_SAFE_RATIO,
        _QUERY_MAJOR_OUTPUT if query_major_output is None else query_major_output, _AGGREGATE_SCATTER)
    return (values, slopes) if return_grad else values
