import pytest
import torch

import torchbvh
import torchbvh._mls as mls_module


def _make_batched_cloud(batch_size: int = 2, n: int = 24, dim: int = 3):
    base = torch.linspace(-1.0, 1.0, n, device="cuda", dtype=torch.float32)
    samples = []
    for batch in range(batch_size):
        if dim == 2:
            pts = torch.stack((base + batch * 0.1, base.square() - 0.25), dim=1)
        else:
            pts = torch.stack((base + batch * 0.1, base.square() - 0.25, torch.sin(2.0 * base)), dim=1)
        samples.append(pts)
    points = torch.stack(samples, dim=0).contiguous()
    features = torch.stack(
        (
            points[..., 0] + 0.5 * points[..., 1],
            points[..., 0].square() - points[..., -1],
        ),
        dim=-1,
    ).contiguous()
    return points, features


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("k", [4, 8])
def test_batched_mls_matches_single_sample_loop(dim, k):
    assert torch.cuda.is_available()
    points, features = _make_batched_cloud(dim=dim)
    displaced = (points[:, 3:11, :] + 0.02).contiguous()

    actual = torchbvh.bvh_mls_interpolate_batched(points, displaced, features, k=k)
    expected = torch.stack(
        [
            torchbvh.bvh_mls_interpolate(
                points[batch].contiguous(),
                displaced[batch].contiguous(),
                features[batch].contiguous(),
                k=k,
            )
            for batch in range(points.size(0))
        ],
        dim=0,
    )

    torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)


def test_public_mls_interpolate_reaches_fused_cuda_mls_boundary(monkeypatch):
    assert torch.cuda.is_available()
    points, features = _make_batched_cloud(batch_size=2, n=12, dim=3)
    displaced = (points[:, 2:7, :] + 0.02).contiguous()
    calls = []

    def fake_batched_query(points_arg, displaced_arg, k_arg):
        batch_size, query_count, dim = displaced_arg.shape
        k = int(k_arg)
        calls.append(("query", points_arg.shape, displaced_arg.shape, k))
        indices = torch.zeros((batch_size, query_count, k), device=points_arg.device, dtype=torch.int64)
        distances = torch.zeros((batch_size, query_count, k), device=points_arg.device, dtype=torch.float32)
        positions = torch.zeros((batch_size, query_count, k, dim), device=points_arg.device, dtype=torch.float32)
        return indices, distances, positions

    def fake_fused_forward(
        displaced_arg,
        neighbor_positions,
        indices,
        squared_distances,
        features_arg,
        feature_batch=None,
        *,
        return_grad,
        return_aux=False,
    ):
        calls.append(
            (
                "fused",
                displaced_arg.shape,
                neighbor_positions.shape,
                indices.shape,
                squared_distances.shape,
                features_arg.shape,
                None if feature_batch is None else feature_batch.shape,
                return_grad,
                return_aux,
            )
        )
        output = torch.zeros(
            (displaced_arg.size(0), features_arg.size(-1)),
            device=features_arg.device,
            dtype=features_arg.dtype,
        )
        if return_aux:
            dim = displaced_arg.size(1)
            return (
                output,
                torch.zeros((displaced_arg.size(0), dim, features_arg.size(-1)), device=features_arg.device),
                torch.zeros((displaced_arg.size(0), dim + 1, dim + 1), device=features_arg.device),
                torch.zeros((displaced_arg.size(0),), device=features_arg.device),
            )
        if return_grad:
            return output, torch.zeros(
                (displaced_arg.size(0), displaced_arg.size(1), features_arg.size(-1)),
                device=features_arg.device,
            )
        return output

    monkeypatch.setattr(mls_module.BatchedBVHQuery, "apply", staticmethod(fake_batched_query))
    monkeypatch.setattr(mls_module, "_linear_mls_fused_forward", fake_fused_forward)

    result = torchbvh.mls_interpolate(points, displaced, features, k=8)

    assert result.shape == (2, 5, 2)
    assert [call[0] for call in calls] == ["query", "fused"]
    assert calls[0][1:] == (points.shape, displaced.shape, 8)
    assert calls[1][1] == (10, 3)
    assert calls[1][2] == (10, 8, 3)
    assert calls[1][3] == (10, 8)
    assert calls[1][4] == (10, 8)
    assert calls[1][5] == features.shape
    assert calls[1][6] == (10,)
    assert calls[1][7] is False
    assert calls[1][8] is False


def test_batched_mls_gradient_boundary():
    assert torch.cuda.is_available()
    points, features = _make_batched_cloud()
    points = points.detach().requires_grad_()
    displaced = (points.detach()[:, 3:11, :] + 0.02).contiguous().requires_grad_()
    features = features.detach().requires_grad_()

    interpolated, field_gradient = torchbvh.bvh_mls_interpolate_batched(
        points,
        displaced,
        features,
        k=8,
        return_grad=True,
    )
    loss = interpolated.square().sum() + 0.01 * field_gradient.square().sum()
    loss.backward()

    assert points.grad is None
    assert displaced.grad is not None
    assert features.grad is not None
    assert torch.isfinite(displaced.grad).all()
    assert torch.isfinite(features.grad).all()


def test_batched_mls_rejects_non_contiguous_inputs():
    assert torch.cuda.is_available()
    points, features = _make_batched_cloud()
    displaced = (points[:, 3:11, :] + 0.02).contiguous()
    padded = torch.empty((*points.shape[:-1], points.shape[-1] + 1), device=points.device, dtype=points.dtype)
    padded[..., : points.shape[-1]] = points
    points_non_contiguous = padded[..., : points.shape[-1]]

    with pytest.raises(ValueError, match="contiguous"):
        torchbvh.bvh_mls_interpolate_batched(points_non_contiguous, displaced, features)
