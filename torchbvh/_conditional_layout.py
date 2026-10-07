import torch
from torch.autograd.function import once_differentiable

from . import _C


class _PackRoutedQueries(torch.autograd.Function):
    @staticmethod
    def forward(ctx, true_queries, false_queries, mask):
        ctx.set_materialize_grads(False)
        selected, routes = _C.pack_routed_queries(true_queries, false_queries, mask)
        ctx.save_for_backward(mask)
        return selected, routes

    @staticmethod
    @once_differentiable
    def backward(ctx, d_selected, _):
        if d_selected is None:
            return None, None, None
        (mask,) = ctx.saved_tensors
        true_gradient, false_gradient = _C.unpack_routed_gradient(
            d_selected.contiguous(), mask, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        return (true_gradient if ctx.needs_input_grad[0] else None,
                false_gradient if ctx.needs_input_grad[1] else None, None)


def _pack_routed_queries(true_queries, false_queries, mask):
    return _PackRoutedQueries.apply(true_queries, false_queries, mask)
