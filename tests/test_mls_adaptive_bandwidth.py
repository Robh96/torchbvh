"""Differentiate adaptive MLS, holding only discrete neighbour choices fixed."""
import pytest
import torch
import torchbvh
from torchbvh import _mls, _mls_geometry, _mls_routed


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA MLS")


def reference(points, queries, features, k, bandwidth_min=1e-12, freeze=False):
    # Sort detached distances only to choose integer indices. Selected distances
    # and the lower-median bandwidth are recomputed from live query coordinates.
    indices = (queries.detach()[:, None] - points[None]).square().sum(-1).argsort(-1)[:, :k]
    delta = queries[:, None] - points[indices]
    distance = delta.square().sum(-1)
    median = distance[:, (k - 1) // 2]
    bandwidth = torch.where(median > bandwidth_min, median, bandwidth_min)
    if freeze:
        bandwidth = bandwidth.detach()
    weights = torch.exp(-distance / (2 * bandwidth[:, None]))
    basis = torch.cat((torch.ones_like(delta[..., :1]), delta), -1)
    matrix = basis.transpose(-1, -2) @ (weights[..., None] * basis)
    matrix = matrix + 1e-6 * torch.eye(points.size(-1) + 1, dtype=points.dtype, device=points.device)
    rhs = basis.transpose(-1, -2) @ (weights[..., None] * features[indices])
    coefficients = torch.linalg.solve(matrix, rhs)
    return coefficients[:, 0], coefficients[:, 1:]


@pytest.mark.parametrize("dim,k,channels,count", [
    (dim, k, channels, 3) for dim in (2, 3) for k in (4, 8, 16) for channels in (3, 64)
] + [(2, 4, 4, 4097), (3, 4, 64, 4097)])
@pytest.mark.parametrize("slopes_only", [False, True])
def test_adaptive_query_and_feature_derivatives(dim, k, channels, count, slopes_only, monkeypatch):
    generator = torch.Generator().manual_seed(811 + k + dim)
    points = torch.rand(32, dim, generator=generator).cuda().requires_grad_()
    queries = torch.full((count, dim), .431, device="cuda", requires_grad=True)
    features = torch.randn(32, channels, generator=generator).cuda().requires_grad_()
    calls = []
    expected_route = "mls_geometry_forward" if count >= 4096 else "mls_packed_indexed_forward"
    native = getattr(torchbvh._C, expected_route)

    def audit(*args):
        calls.append(expected_route)
        return native(*args)

    monkeypatch.setattr(torchbvh._C, expected_route, audit)
    outputs = torchbvh.mls_interpolate(points, queries, features, k=k, return_grad=True)
    rq = queries[:1].detach().cpu().double().requires_grad_()
    rf = features.detach().cpu().double().requires_grad_()
    expected = reference(points.detach().cpu().double(), rq, rf, k)
    upstream = [torch.linspace(.01, .1, channels, dtype=torch.float64),
                torch.linspace(-.02, .03, dim * channels, dtype=torch.float64).reshape(dim, channels)]
    actual_loss = sum((o[0] * g.to(o)).sum() for i, (o, g) in enumerate(zip(outputs, upstream))
                      if not slopes_only or i == 1)
    ref_loss = sum((o[0] * g).sum() for i, (o, g) in enumerate(zip(expected, upstream))
                   if not slopes_only or i == 1)
    actual_loss.backward()
    ref_loss.backward()
    for actual, target in zip(outputs, expected):
        torch.testing.assert_close(actual[0].cpu().double(), target[0], rtol=8e-4, atol=2e-4)
    torch.testing.assert_close(queries.grad[0].cpu().double(), rq.grad[0], rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(features.grad.cpu().double(), rf.grad, rtol=3e-3, atol=3e-3)
    assert not bool(queries.grad[1:].any())
    assert points.grad is None
    assert calls == [expected_route]


@pytest.mark.parametrize("channels,count", [(1, 1), (64, 1), (4, 4097)])
@pytest.mark.parametrize("bandwidth_min", [1e-12, 10.0])
def test_non_affine_finite_difference_and_bandwidth_floor(channels, count, bandwidth_min, monkeypatch):
    # Known counterexample: affine fields can conceal the missing chain rule.
    for module in (_mls, _mls_geometry, _mls_routed):
        monkeypatch.setattr(module, "MLS_BANDWIDTH_MIN", bandwidth_min)
    points = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.],
                           [.5, .1], [.2, .4], [.8, .7], [.3, .9]], device="cuda")
    features = torch.tensor([0., 1., 2., 4., 3., 6., 1., 2.], device="cuda")[:, None].repeat(1, channels)
    queries = torch.tensor([[.43, .37]], device="cuda").repeat(count, 1).requires_grad_()
    actual = torchbvh.mls_interpolate(points, queries, features, k=4)
    actual[0].mean().backward()
    rq = queries[:1].detach().cpu().double().requires_grad_()
    rp, rf = points.cpu().double(), features[:, :1].cpu().double()
    expected, _ = reference(rp, rq, rf, 4, bandwidth_min)
    expected.sum().backward()
    step = 1e-5
    finite_difference = []
    for axis in range(2):
        shift = torch.zeros_like(rq)
        shift[0, axis] = step
        plus = reference(rp, rq.detach() + shift, rf, 4, bandwidth_min)[0]
        minus = reference(rp, rq.detach() - shift, rf, 4, bandwidth_min)[0]
        finite_difference.append(((plus - minus) / (2 * step)).item())
    torch.testing.assert_close(rq.grad[0], torch.tensor(finite_difference, dtype=torch.float64), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual[0, 0].cpu().double(), expected[0, 0], rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(queries.grad[0].cpu().double(), rq.grad[0], rtol=3e-4, atol=1e-3)
    frozen_q = rq.detach().requires_grad_()
    reference(rp, frozen_q, rf, 4, bandwidth_min, freeze=True)[0].sum().backward()
    if bandwidth_min == 10.0:
        torch.testing.assert_close(rq.grad, frozen_q.grad, rtol=0, atol=0)
    else:
        assert abs((rq.grad - frozen_q.grad)[0, 1]) > .5


@pytest.mark.parametrize("channels,count", [(1, 1), (64, 1), (4, 4097)])
def test_bandwidth_floor_boundary_uses_zero_bandwidth_derivative(channels, count, monkeypatch):
    # Binary-exact coordinates/distances avoid rounding across the clamp boundary.
    for module in (_mls, _mls_geometry, _mls_routed):
        monkeypatch.setattr(module, "MLS_BANDWIDTH_MIN", .3125)
    points = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], device="cuda")
    queries = torch.tensor([[.5, .25]], device="cuda").repeat(count, 1).requires_grad_()
    features = torch.tensor([[0.], [1.], [2.], [4.]], device="cuda").repeat(1, channels)
    values, slopes = torchbvh.mls_interpolate(points, queries, features, k=4, return_grad=True)
    (values[0].mean() + .03 * slopes[0].sum() / channels).backward()
    rq = queries[:1].detach().cpu().double().requires_grad_()
    rv, rs = reference(points.cpu().double(), rq, features[:, :1].cpu().double(), 4, .3125)
    (rv.sum() + .03 * rs.sum()).backward()
    torch.testing.assert_close(queries.grad[0].cpu().double(), rq.grad[0], rtol=3e-4, atol=1e-3)


@pytest.mark.parametrize("count", [3, 4097])
def test_conditional_adaptive_bandwidth_routes_both_banks(count):
    generator = torch.Generator().manual_seed(414)
    points = torch.rand(2, 24, 2, generator=generator).cuda()
    other = points + 2
    queries = torch.full((2, count, 2, 2), .431, device="cuda", requires_grad=True)
    other_queries = (queries.detach() + 2).requires_grad_()
    features = torch.randn(2, 24, 2, 4, generator=generator).cuda().requires_grad_()
    other_features = (features.detach() * .7).requires_grad_()
    mask = (torch.arange(count, device="cuda")[None, :, None] % 2 == 0).expand(2, -1, 2).contiguous()
    with torchbvh.PointGeometry(points) as geometry, torchbvh.PointGeometry(other) as other_geometry:
        values, slopes = torchbvh.conditional_mls_interpolate(
            mask, true_points=points, true_queries=queries, true_features=features, true_geometry=geometry,
            false_points=other, false_queries=other_queries, false_features=other_features,
            false_geometry=other_geometry, k=4, return_grad=True)
    (values[:, :2].sum() + .03 * slopes[:, :2].sum()).backward()
    for batch in range(2):
        for row, bank in ((0, (points, queries, features)), (1, (other, other_queries, other_features))):
            for head in range(2):
                p, q, f = bank
                rq = q[batch, row, head].detach().cpu().double()[None].requires_grad_()
                rf = f[batch, :, head].detach().cpu().double().requires_grad_()
                rv, rs = reference(p[batch].cpu().double(), rq, rf, 4)
                (rv.sum() + .03 * rs.sum()).backward()
                torch.testing.assert_close(q.grad[batch, row, head].cpu().double(), rq.grad[0], rtol=3e-3, atol=3e-3)
                torch.testing.assert_close(f.grad[batch, :, head].cpu().double(), rf.grad, rtol=3e-3, atol=3e-3)
    assert not bool(other_queries.grad[:, 0].any())
    assert not bool(queries.grad[:, 1].any())
