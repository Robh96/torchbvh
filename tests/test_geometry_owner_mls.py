"""Automatic geometry-owner dispatch must match the retained MLS fallback."""
import pytest
import torch
import torchbvh
from torchbvh import _mls_geometry


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("channels", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("return_grad", [False, True])
@pytest.mark.parametrize("requested", ["both", "queries", "features"])
@pytest.mark.legacy_gradient_parity
def test_geometry_owner_ordinary_values_and_gradients(dim, channels, return_grad, requested, monkeypatch):
    torch.manual_seed(931 + channels + dim)
    points = torch.rand(1, 65, dim, device="cuda")
    queries = torch.rand(1, 4097, dim, device="cuda", requires_grad=requested != "features")
    features = torch.randn(1, 65, channels, device="cuda", requires_grad=requested != "queries")
    # Exact hits plus a final partial CUDA block exercise both state branches.
    with torch.no_grad():
        queries[:, :3].copy_(points[:, :3])
    value_gradients = torch.randn(1, 4097, channels, device="cuda")
    slope_gradients = torch.randn(1, 4097, dim, channels, device="cuda") if return_grad else None
    enabled = _mls_geometry.eligible
    native = torchbvh._C.mls_geometry_forward
    calls = []

    def audit(*args):
        result = native(*args)
        calls.append(result[2].shape)
        return result

    monkeypatch.setattr(torchbvh._C, "mls_geometry_forward", audit)

    def evaluate(use_geometry):
        monkeypatch.setattr(_mls_geometry, "eligible", enabled if use_geometry else lambda *args: False)
        queries.grad = features.grad = None
        outputs = torchbvh.mls_interpolate(points, queries, features, k=4, return_grad=return_grad)
        values, slopes = outputs if return_grad else (outputs, None)
        loss = (values * value_gradients).sum()
        if return_grad:
            loss = loss + (slopes * slope_gradients).sum()
        loss.backward()
        return ([t.detach().clone() for t in (values, slopes) if t is not None],
                [t.grad.clone() for t in (queries, features) if t.requires_grad])

    expected, actual = evaluate(False), evaluate(True)
    assert calls == [torch.Size([4097, 7 if dim == 2 else 11])]
    for a, b in zip(actual[0], expected[0]):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
    for a, b in zip(actual[1], expected[1]):
        torch.testing.assert_close(a, b, rtol=5e-5, atol=5e-5)


@pytest.mark.legacy_gradient_parity
def test_geometry_owner_matching_heads_and_geometry_lifetime(monkeypatch):
    torch.manual_seed(703)
    points = torch.rand(2, 81, 3, device="cuda")
    queries = torch.rand(2, 1025, 2, 3, device="cuda", requires_grad=True)
    features = torch.randn(2, 81, 2, 64, device="cuda", requires_grad=True)
    enabled = _mls_geometry.eligible
    monkeypatch.setattr(_mls_geometry, "eligible", lambda *args: False)
    expected = torchbvh.bvh_mls_interpolate_batched_heads(points, queries, features, k=4)
    reference = torch.autograd.grad(expected.square().sum(), (queries, features))
    monkeypatch.setattr(_mls_geometry, "eligible", enabled)
    with torchbvh.PointGeometry(points) as geometry:
        actual = torchbvh.bvh_mls_interpolate_batched_heads(points, queries, features, k=4, geometry=geometry)
    gradients = torch.autograd.grad(actual.square().sum(), (queries, features))
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    for a, b in zip(gradients, reference):
        torch.testing.assert_close(a, b, rtol=5e-5, atol=5e-5)


@pytest.mark.parametrize("dim,k,channels,queries", [(2, 8, 64, 4096), (3, 16, 16, 4096),
                                                   (2, 4, 7, 4096), (3, 4, 64, 4095)])
def test_geometry_owner_unsupported_shapes_use_fallback(dim, k, channels, queries, monkeypatch):
    def forbidden(*args):
        pytest.fail("Unsupported workload reached geometry-owner native kernel")
    monkeypatch.setattr(torchbvh._C, "mls_geometry_forward", forbidden)
    points = torch.rand(1, 65, dim, device="cuda")
    values = torchbvh.mls_interpolate(points, torch.rand(1, queries, dim, device="cuda"),
                                    torch.randn(1, 65, channels, device="cuda"), k=k)
    assert values.shape == (1, queries, channels) and torch.isfinite(values).all()
