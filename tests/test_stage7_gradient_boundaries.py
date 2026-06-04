import pytest
import torch

import torchbvh


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Stage 7 gradient-boundary validation requires CUDA",
)


def _assert_finite_nonzero_grad(tensor):
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()
    assert tensor.grad.abs().sum() > 0


def _assert_no_grad(tensor):
    assert tensor.grad is None


def _base_points(dim, *, duplicate_heavy=False, degenerate=False):
    n = 16
    if degenerate:
        t = torch.linspace(-1.0, 1.0, n, device="cuda", dtype=torch.float32)
        if dim == 2:
            sample = torch.stack((t, 0.25 * t), dim=-1)
        else:
            sample = torch.stack((t, 0.25 * t, torch.zeros_like(t)), dim=-1)
    elif duplicate_heavy:
        sample = torch.zeros((n, dim), device="cuda", dtype=torch.float32)
        t = torch.linspace(0.1, 1.0, n - 8, device="cuda", dtype=torch.float32)
        sample[8:, 0] = t
        if dim >= 2:
            sample[8:, 1] = t.square()
        if dim == 3:
            sample[8:, 2] = 0.5 * t
    else:
        t = torch.linspace(-0.75, 0.85, n, device="cuda", dtype=torch.float32)
        if dim == 2:
            sample = torch.stack((t, torch.sin(2.0 * t)), dim=-1)
        else:
            sample = torch.stack((t, torch.sin(2.0 * t), torch.cos(1.5 * t)), dim=-1)

    offsets = torch.arange(2, device="cuda", dtype=torch.float32).view(2, 1, 1) * 5.0
    return (sample.unsqueeze(0) + offsets).contiguous()


def _feature_tensor(points, channels):
    cols = [points[..., 0]]
    if points.size(-1) > 1:
        cols.append(points[..., 1])
    cols.append(points[..., 0].square() + 0.25)
    while len(cols) < channels:
        cols.append(torch.sin((len(cols) + 1) * points[..., 0]) + 0.1 * len(cols))
    return torch.stack(cols[:channels], dim=-1).contiguous()


@pytest.mark.parametrize(
    ("name", "dim", "k", "duplicate_heavy", "degenerate"),
    [
        ("exact_hit", 2, 4, False, False),
        ("duplicate_heavy", 3, 8, True, False),
        ("degenerate_underdetermined", 3, 4, False, True),
    ],
)
def test_stage7_batched_mls_training_composition_gradient_boundaries(
    name,
    dim,
    k,
    duplicate_heavy,
    degenerate,
):
    points = _base_points(dim, duplicate_heavy=duplicate_heavy, degenerate=degenerate).requires_grad_(True)
    features = _feature_tensor(points.detach(), channels=4).requires_grad_(True)
    offsets = torch.zeros((2, 6, dim), device="cuda", dtype=torch.float32)
    offsets[:, 1:, 0] = torch.linspace(0.01, 0.05, 5, device="cuda")
    if dim == 3:
        offsets[:, 2:, 2] = 0.015
    if name == "exact_hit":
        offsets[:, 0] = 0.0
    displaced_points = (points.detach()[:, :6, :] + offsets).contiguous().requires_grad_(True)

    interpolated, field_gradient = torchbvh.bvh_mls_interpolate_batched(
        points,
        displaced_points,
        features,
        k=k,
        return_grad=True,
    )

    assert interpolated.shape == (2, 6, 4)
    assert field_gradient.shape == (2, 6, dim, 4)
    assert torch.isfinite(interpolated).all()
    assert torch.isfinite(field_gradient).all()
    (interpolated.square().mean() + 0.1 * field_gradient.square().mean()).backward()

    _assert_finite_nonzero_grad(features)
    _assert_finite_nonzero_grad(displaced_points)
    _assert_no_grad(points)


@pytest.mark.parametrize(
    ("dim", "k", "duplicate_heavy"),
    [
        (2, 4, False),
        (3, 8, True),
    ],
)
def test_stage7_displaced_query_interpolation_values_only_gradient_boundary(dim, k, duplicate_heavy):
    pos = _base_points(dim, duplicate_heavy=duplicate_heavy, degenerate=False).requires_grad_(True)
    rho_data = torch.zeros((2, pos.size(1), 3, dim), device="cuda", dtype=torch.float32)
    rho_data[:, :, 1, 0] = 0.02
    rho_data[:, :, 2, 0] = torch.linspace(-0.03, 0.03, pos.size(1), device="cuda")
    if dim == 3:
        rho_data[:, :, 2, 2] = 0.01
    rho = rho_data.contiguous().requires_grad_(True)
    q = (pos[:, :, None, :] + rho).contiguous()
    q.retain_grad()
    values = _feature_tensor(pos.detach(), channels=5).unsqueeze(2).repeat(1, 1, 3, 1).contiguous()
    values = values.requires_grad_(True)

    output = torchbvh.interpolate_displaced(pos, q, values, k)
    indices, squared_distances, neighbor_positions = torchbvh.query_displaced_knn(pos, q, k)

    assert output.shape == values.shape
    assert torch.isfinite(output).all()
    assert (squared_distances[:, :, 0] <= 1.0e-12).any()
    assert not indices.requires_grad
    assert not squared_distances.requires_grad
    assert not neighbor_positions.requires_grad
    output.square().mean().backward()

    _assert_finite_nonzero_grad(values)
    _assert_no_grad(pos)
    _assert_no_grad(rho)
    _assert_no_grad(q)

