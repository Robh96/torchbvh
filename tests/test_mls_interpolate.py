import pytest
import torch

import torchbvh


def _make_cloud(dim: int):
    base = torch.linspace(-1.0, 1.0, 24, device="cuda", dtype=torch.float32)
    if dim == 2:
        points = torch.stack((base, base.square() - 0.25), dim=1)
    else:
        points = torch.stack((base, base.square() - 0.25, torch.sin(2.0 * base)), dim=1)
    features = torch.stack(
        (
            points[:, 0] + 0.5 * points[:, 1],
            points[:, 0].square() - points[:, -1],
        ),
        dim=1,
    ).contiguous()
    return points.contiguous(), features


def _make_batched_cloud(dim: int):
    points, features = _make_cloud(dim)
    points = torch.stack((points, points + 0.13), dim=0).contiguous()
    features = torch.stack((features, features * 0.75 + 0.2), dim=0).contiguous()
    return points, features


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("k", [4, 8])
def test_bvh_mls_interpolate_public_shapes_and_gradients(dim, k):
    assert torch.cuda.is_available()
    points, features = _make_cloud(dim)
    displaced_points = (points[3:11] + 0.02).contiguous().requires_grad_()
    features = features.clone().requires_grad_()

    interpolated, field_gradient = torchbvh.mls_interpolate(
        points,
        displaced_points,
        features,
        k=k,
        return_grad=True,
    )
    loss = interpolated.square().sum() + 0.01 * field_gradient.square().sum()
    loss.backward()

    assert interpolated.shape == (8, features.shape[-1])
    assert field_gradient.shape == (8, dim, features.shape[-1])
    assert torch.isfinite(interpolated).all()
    assert torch.isfinite(field_gradient).all()
    assert features.grad is not None
    assert displaced_points.grad is not None


def test_bvh_mls_interpolate_batched_default_returns_tensor_not_tuple():
    assert torch.cuda.is_available()
    points, features = _make_batched_cloud(3)
    displaced_points = (points[:, 3:7, :] + 0.02).contiguous()

    result = torchbvh.mls_interpolate(points, displaced_points, features)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 4, features.shape[-1])


def test_bvh_mls_interpolate_batched_return_grad_true_returns_tuple_with_correct_shapes():
    assert torch.cuda.is_available()
    points, features = _make_batched_cloud(3)
    displaced_points = (points[:, 3:7, :] + 0.02).contiguous()

    result = torchbvh.mls_interpolate(points, displaced_points, features, k=8, return_grad=True)

    assert isinstance(result, tuple)
    assert len(result) == 2
    interpolated, field_gradient = result
    assert interpolated.shape == (2, 4, features.shape[-1])
    assert field_gradient.shape == (2, 4, 3, features.shape[-1])


@pytest.mark.parametrize("dim,k", [(2, 4), (3, 8)])
def test_bvh_mls_interpolate_batched_heads_matches_per_head_loop_and_backpropagates(dim, k):
    assert torch.cuda.is_available()
    torch.manual_seed(314)
    batch_size, num_points, num_queries, num_heads, channels = 2, 32, 9, 3, 4
    points = torch.randn(
        (batch_size, num_points, dim), device="cuda", dtype=torch.float32
    ).contiguous()
    queries = (
        points[:, :num_queries, :].unsqueeze(2)
        + 0.015
        * torch.randn(
            (batch_size, num_queries, num_heads, dim),
            device="cuda",
            dtype=torch.float32,
        )
    ).contiguous().requires_grad_()
    features = torch.randn(
        (batch_size, num_points, num_heads, channels),
        device="cuda",
        dtype=torch.float32,
    ).contiguous().requires_grad_()

    expected = torch.stack(
        [
            torchbvh.mls_interpolate(
                points,
                queries[:, :, head],
                features[:, :, head],
                k,
            )
            for head in range(num_heads)
        ],
        dim=2,
    )
    actual = torchbvh.bvh_mls_interpolate_batched_heads(
        points,
        queries,
        features,
        k,
    )

    torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)

    actual.square().sum().backward()

    assert queries.grad is not None
    assert features.grad is not None
