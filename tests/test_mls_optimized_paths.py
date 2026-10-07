"""Scheduling and output specialization must preserve MLS mathematics."""
from contextlib import contextmanager

import pytest
import torch
import torchbvh


@contextmanager
def _ordering(bits):
    original = torchbvh._C.morton_sort_routed_queries_batched
    torchbvh._C.morton_sort_routed_queries_batched = lambda *args: torchbvh._C.morton_sort_routed_queries_narrow(*args, bits)
    try:
        yield
    finally:
        torchbvh._C.morton_sort_routed_queries_batched = original


@pytest.mark.parametrize("dim,k,channels", [(2, 4, 4), (2, 8, 8), (3, 16, 16), (3, 4, 64)])
def test_values_only_and_narrow_sort_preserve_values_slopes_and_backward(dim, k, channels):
    torch.manual_seed(429)
    batch, queries, heads = 2, 29, 3
    mask = torch.rand(batch, queries, heads, device="cuda") > .4
    kwargs = {}
    for route, count in (("true", 37), ("false", 71)):
        points = torch.rand(batch, count, dim, device="cuda")
        points[:, :4] = points[:, :1]  # duplicate exact hits
        q = torch.rand(batch, queries, heads, dim, device="cuda")
        q[:, 0] = points[:, :1]
        q[~mask if route == "true" else mask] = float("nan")
        kwargs[route + "_points"] = points
        kwargs[route + "_queries"] = q.requires_grad_()
        kwargs[route + "_features"] = torch.randn(batch, count, heads, channels, device="cuda", requires_grad=True)
    upstream = torch.randn(batch, queries, heads, channels, device="cuda")
    tensors = [v for v in kwargs.values() if v.requires_grad]
    def evaluate(return_grad):
        for tensor in tensors:
            tensor.grad = None
        result = torchbvh.conditional_mls_interpolate(mask, **kwargs, k=k, return_grad=return_grad)
        values = result[0] if return_grad else result
        values.backward(upstream)
        return values.detach(), result[1].detach() if return_grad else None, [t.grad.clone() for t in tensors]
    reference = evaluate(True)
    for bits in (30, 24, 12):
        with _ordering(bits):
            for return_grad in (False, True):
                actual = evaluate(return_grad)
                torch.testing.assert_close(actual[0], reference[0], atol=2e-5, rtol=2e-5)
                if return_grad:
                    torch.testing.assert_close(actual[1], reference[1], atol=2e-5, rtol=2e-5)
                for a, r in zip(actual[2], reference[2]):
                    torch.testing.assert_close(a, r, atol=5e-5, rtol=5e-5)
