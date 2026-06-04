import pytest
import torch

import torchbvh
from torchbvh._mls import (
    _bvh_mls_interpolate_batched_head_banked,
    _linear_mls_chunk,
    _linear_mls_fused_forward,
)


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

    interpolated, field_gradient = torchbvh.bvh_mls_interpolate(
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

    result = torchbvh.bvh_mls_interpolate_batched(points, displaced_points, features)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 4, features.shape[-1])


def test_bvh_mls_interpolate_batched_return_grad_true_returns_tuple_with_correct_shapes():
    assert torch.cuda.is_available()
    points, features = _make_batched_cloud(3)
    displaced_points = (points[:, 3:7, :] + 0.02).contiguous()

    result = torchbvh.bvh_mls_interpolate_batched(points, displaced_points, features, k=8, return_grad=True)

    assert isinstance(result, tuple)
    assert len(result) == 2
    interpolated, field_gradient = result
    assert interpolated.shape == (2, 4, features.shape[-1])
    assert field_gradient.shape == (2, 4, 3, features.shape[-1])


def test_fused_forward_matches_reference_chunk_on_same_neighbors():
    assert torch.cuda.is_available()
    torch.manual_seed(123)
    points = torch.randn((32, 3), device="cuda", dtype=torch.float32).contiguous()
    displaced_points = (points[:12] + 0.03 * torch.randn((12, 3), device="cuda")).contiguous()
    features = torch.randn((32, 5), device="cuda", dtype=torch.float32).contiguous()
    bvh = torchbvh.build_bvh(points)
    indices, squared_distances, neighbor_positions = torchbvh.query_knn(
        bvh,
        displaced_points,
        8,
        source_points=points,
    )
    neighbor_features = features[indices]

    expected, expected_gradient = _linear_mls_chunk(
        displaced_points,
        neighbor_positions,
        neighbor_features,
        squared_distances,
        return_grad=True,
    )
    actual, actual_gradient = _linear_mls_fused_forward(
        displaced_points,
        neighbor_positions,
        indices,
        squared_distances,
        features,
        return_grad=True,
    )

    torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=1.0e-2)
    assert torch.isfinite(actual_gradient).all()
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0.25, atol=0.25)


def test_private_head_banked_batched_mls_matches_public_per_head_loop():
    assert torch.cuda.is_available()
    torch.manual_seed(314)
    B, N, H, M, D, C_head = 2, 32, 3, 9, 3, 4
    points = torch.randn((B, N, D), device="cuda", dtype=torch.float32).contiguous()
    displaced_by_head = (
        points[:, :M, :].unsqueeze(1)
        + 0.015 * torch.randn((B, H, M, D), device="cuda", dtype=torch.float32)
    ).contiguous()
    features_by_head = torch.randn((B, N, H, C_head), device="cuda", dtype=torch.float32).contiguous()

    packed = _bvh_mls_interpolate_batched_head_banked(points, displaced_by_head, features_by_head, k=8)
    per_head = torch.stack(
        [
            torchbvh.bvh_mls_interpolate_batched(
                points,
                displaced_by_head[:, head, :, :].contiguous(),
                features_by_head[:, :, head, :].contiguous(),
                k=8,
            )
            for head in range(H)
        ],
        dim=2,
    )

    torch.testing.assert_close(packed, per_head, rtol=2.0e-5, atol=2.0e-5)
