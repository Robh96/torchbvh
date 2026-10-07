"""Prepared geometry shares construction, never source or autograd lifetime."""

import pytest
import torch
import torchbvh as tb


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("dim", [2, 3])
def test_prepared_matches_values_gradients_and_survives_close(batched, dim):
    torch.manual_seed(719)
    shape = (2, 31, dim) if batched else (31, dim)
    # Non-contiguous sources exercise identity checks before normalization.
    points = torch.rand((*shape[:-1], dim + 1), device="cuda")[..., :dim]
    queries = torch.rand((*shape[:-2], 13, dim), device="cuda", requires_grad=True)
    features = torch.rand((*shape[:-1], 7), device="cuda", requires_grad=True)
    expected, slope = tb.mls_interpolate(points, queries, features, return_grad=True)
    reference = torch.autograd.grad(expected.square().sum() + .03*slope.square().sum(), (queries, features))
    with tb.PointGeometry(points) as geometry:
        actual, actual_slope = tb.mls_interpolate(points, queries, features, return_grad=True, geometry=geometry)
    # Source storage changes after the forward must not change its backward.
    points.add_(2)
    grads = torch.autograd.grad(actual.square().sum() + .03*actual_slope.square().sum(), (queries, features))
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual_slope, slope, rtol=2e-5, atol=2e-5)
    for a, b in zip(grads, reference):
        torch.testing.assert_close(a, b, rtol=5e-5, atol=5e-5)


def test_geometry_rejects_other_source_mutation_and_closed_handles():
    points = torch.rand(2, 32, 2, device="cuda")
    geometry = tb.PointGeometry(points)
    with pytest.raises(ValueError, match="different source"):
        geometry.validate(points.clone())
    points.add_(1)
    with pytest.raises(ValueError, match="source changed"):
        geometry.validate(points)
    geometry.destroy()
    geometry.destroy()
    with pytest.raises(RuntimeError, match="destroyed"):
        geometry.validate(points)


def test_conditional_geometry_reuses_two_builds_and_routes_gradients(monkeypatch):
    torch.manual_seed(443)
    from torchbvh import _geometry
    builds = []
    build = _geometry._build_bvh_batched
    def count(points):
        builds.append(tuple(points.shape))
        return build(points)
    monkeypatch.setattr(_geometry, "_build_bvh_batched", count)
    points = [torch.rand(2, n, 2, device="cuda") for n in (19, 37)]
    queries = [torch.rand(2, 11, 3, 2, device="cuda", requires_grad=True) for _ in range(2)]
    features = [torch.rand(2, n, 3, 4, device="cuda", requires_grad=True) for n in (19, 37)]
    mask = torch.rand(2, 11, 3, device="cuda") > .4
    kw = dict(true_points=points[0], false_points=points[1], true_queries=queries[0], false_queries=queries[1], true_features=features[0], false_features=features[1])
    ref = tb.conditional_mls_interpolate(mask, **kw)
    with tb.PointGeometry(points[0]) as a, tb.PointGeometry(points[1]) as b:
        outputs = [tb.conditional_mls_interpolate(mask, **kw, true_geometry=a, false_geometry=b) for _ in range(3)]
    assert len(builds) == 2
    torch.testing.assert_close(outputs[0], ref, rtol=2e-5, atol=2e-5)
    outputs[0].sum().backward()
    assert torch.count_nonzero(queries[0].grad[~mask]) == 0
    assert torch.count_nonzero(queries[1].grad[mask]) == 0


def test_multihead_geometry_matches_unprepared():
    p = torch.rand(2, 32, 3, device="cuda")
    q = torch.rand(2, 11, 4, 3, device="cuda")
    f = torch.rand(2, 32, 4, 5, device="cuda")
    expected = tb.bvh_mls_interpolate_batched_heads(p, q, f)
    with tb.PointGeometry(p) as geometry:
        actual = tb.bvh_mls_interpolate_batched_heads(p, q, f, geometry=geometry)
    torch.testing.assert_close(actual, expected)
