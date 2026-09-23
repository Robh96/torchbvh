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

    actual = torchbvh.mls_interpolate(points, displaced, features, k=k)
    expected = torch.stack(
        [
            torchbvh.mls_interpolate(
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


def test_public_mls_interpolate_reaches_spatial_indexed_mls_boundary(monkeypatch):
    assert torch.cuda.is_available()
    points, features = _make_batched_cloud(batch_size=2, n=12, dim=3)
    displaced = (points[:, 2:7, :] + 0.02).contiguous()
    calls = []

    def fake_spatial_indexed(
        displaced_arg,
        source_points,
        indices,
        squared_distances,
        feature_bank,
        query_order,
        *,
        queries_per_batch,
        queries_per_head,
        return_grad,
    ):
        calls.append(
            (
                "spatial_indexed",
                displaced_arg.shape,
                source_points.shape,
                indices.shape,
                squared_distances.shape,
                feature_bank.shape,
                query_order.shape,
                queries_per_batch,
                queries_per_head,
                return_grad,
            )
        )
        output = torch.zeros(
            (displaced_arg.size(0), feature_bank.size(-1)),
            device=feature_bank.device,
            dtype=feature_bank.dtype,
        )
        if return_grad:
            return output, torch.zeros(
                (displaced_arg.size(0), displaced_arg.size(1), feature_bank.size(-1)),
                device=feature_bank.device,
            )
        return output

    monkeypatch.setattr(mls_module, "_linear_mls_spatial_indexed", fake_spatial_indexed)

    result = torchbvh.mls_interpolate(points, displaced, features, k=8)

    assert result.shape == (2, 5, 2)
    assert [call[0] for call in calls] == ["spatial_indexed"]
    assert calls[0][1] == (10, 3)
    assert calls[0][2] == points.shape
    assert calls[0][3] == (10, 8)
    assert calls[0][4] == (10, 8)
    assert calls[0][5] == (2, 12, 2)
    assert calls[0][6] == (10,)
    assert calls[0][7] == 5
    assert calls[0][8] == 5
    assert calls[0][9] is False


def test_batched_mls_gradient_boundary():
    assert torch.cuda.is_available()
    points, features = _make_batched_cloud()
    points = points.detach().requires_grad_()
    displaced = (points.detach()[:, 3:11, :] + 0.02).contiguous().requires_grad_()
    features = features.detach().requires_grad_()

    interpolated, field_gradient = torchbvh.mls_interpolate(
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


def test_batched_mls_accepts_non_contiguous_inputs():
    assert torch.cuda.is_available()
    points, features = _make_batched_cloud()
    displaced = (points[:, 3:11, :] + 0.02).contiguous()
    padded = torch.empty((*points.shape[:-1], points.shape[-1] + 1), device=points.device, dtype=points.dtype)
    padded[..., : points.shape[-1]] = points
    points_non_contiguous = padded[..., : points.shape[-1]]

    expected = torchbvh.mls_interpolate(points, displaced, features)
    actual = torchbvh.mls_interpolate(points_non_contiguous, displaced, features)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
