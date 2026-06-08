import pytest
import torch

import torchbvh
import torchbvh._multihead as multihead_module


def _displaced_query_inputs(*, batch_size=2, n=24, heads=3, dim=3, channels=4):
    torch.manual_seed(20500 + batch_size * 100 + n * 10 + heads + dim)
    pos = torch.rand((batch_size, n, dim), device="cuda", dtype=torch.float32)
    if batch_size > 1:
        offsets = torch.arange(batch_size, device="cuda", dtype=torch.float32).view(batch_size, 1, 1)
        pos = pos + 8.0 * offsets
    q = pos[:, :, None, :] + 0.015 * torch.randn(
        (batch_size, n, heads, dim),
        device="cuda",
        dtype=torch.float32,
    )
    values = torch.randn((batch_size, n, heads, channels), device="cuda", dtype=torch.float32)
    return pos.contiguous(), q.contiguous(), values.contiguous()


def _robust_displaced_query_inputs(*, dim, k):
    n = 32
    base = torch.zeros((n, dim), device="cuda", dtype=torch.float32)
    base[:4] = 0.0
    t = torch.linspace(-1.0, 1.0, n - 4, device="cuda", dtype=torch.float32)
    if dim == 2:
        rows = torch.stack((t, 0.25 * t), dim=-1)
        rows[4:10] = torch.stack((1.0e-5 * t[:6], torch.zeros(6, device="cuda")), dim=-1)
        rows[10:18] = torch.stack((1.0e4 + 17.0 * t[6:14], -2.0e4 + 3.0 * t[6:14]), dim=-1)
        rows[18:26] = torch.stack((1.0e-3 * t[14:22], 1.0e3 * t[14:22]), dim=-1)
    else:
        rows = torch.stack((t, 0.5 * t, torch.zeros_like(t)), dim=-1)
        rows[4:10] = torch.stack((1.0e-5 * t[:6], torch.zeros(6, device="cuda"), torch.zeros(6, device="cuda")), dim=-1)
        rows[10:18] = torch.stack(
            (1.0e4 + 17.0 * t[6:14], -2.0e4 + 3.0 * t[6:14], 5.0e3 - 11.0 * t[6:14]),
            dim=-1,
        )
        rows[18:26] = torch.stack((1.0e-3 * t[14:22], 1.0e3 * t[14:22], 0.25 * t[14:22]), dim=-1)
    base[4:] = rows

    offsets = torch.arange(2, device="cuda", dtype=torch.float32).view(2, 1, 1) * 7.5e4
    pos = (base.unsqueeze(0) + offsets).contiguous()
    q = pos[:, :, None, :].repeat(1, 1, 3, 1)
    head_offsets = torch.zeros_like(q)
    head_offsets[:, :, 1, :] = 0.013
    head_offsets[:, :, 2, 0] = torch.linspace(-0.02, 0.02, n, device="cuda")
    if dim == 3:
        head_offsets[:, :, 2, 2] = 0.007
    q = (q + head_offsets).contiguous()
    values = torch.arange(2 * n * 3 * 5, device="cuda", dtype=torch.float32).reshape(2, n, 3, 5)
    return pos, q, values.contiguous()


def _old_style_weighted_mean_reference(values, indices, squared_distances):
    neighbor_values = torchbvh.gather_neighbor_values(values, indices)
    exact_mask = squared_distances <= 1.0e-12
    weights = torch.reciprocal(squared_distances + 1.0e-12)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1.0e-12)
    expected = (neighbor_values * weights.unsqueeze(-1)).sum(dim=-2)
    if exact_mask.any():
        exact_weights = exact_mask.to(neighbor_values.dtype).unsqueeze(-1)
        exact_counts = exact_weights.sum(dim=-2).clamp(min=1.0)
        exact_average = (neighbor_values * exact_weights).sum(dim=-2) / exact_counts
        expected = torch.where(exact_mask.any(dim=-1, keepdim=True), exact_average, expected)
    return expected


@pytest.mark.parametrize(
    "bad_pos,bad_q,match",
    [
        (lambda p: p[0], lambda q: q, "pos.*shape"),
        (lambda p: p, lambda q: q.reshape(q.size(0), q.size(1) * q.size(2), q.size(3)), "q.*shape"),
        (lambda p: p[:1], lambda q: q, "batch size"),
        (lambda p: p, lambda q: q[:, :3], "point dimension"),
        (lambda p: p, lambda q: q.double(), "float32"),
    ],
)
def test_query_displaced_knn_shape_validation(bad_pos, bad_q, match):
    assert torch.cuda.is_available()
    pos, q, _ = _displaced_query_inputs()

    with pytest.raises(ValueError, match=match):
        torchbvh.query_displaced_knn(bad_pos(pos), bad_q(q), 4)


def test_query_displaced_knn_returns_local_indices_sorted_distances_and_positions():
    assert torch.cuda.is_available()
    pos, q, _ = _displaced_query_inputs(batch_size=3, n=28, heads=2, dim=2)

    indices, distances, neighbor_positions = torchbvh.query_displaced_knn(pos, q, 4)

    assert indices.shape == (3, 28, 2, 4)
    assert distances.shape == (3, 28, 2, 4)
    assert neighbor_positions.shape == (3, 28, 2, 4, 2)
    assert indices.dtype == torch.int64
    assert distances.dtype == torch.float32
    assert int(indices.min()) >= 0
    assert int(indices.max()) < pos.size(1)
    assert torch.all(distances[:, :, :, :-1] <= distances[:, :, :, 1:])

    expected_positions = torch.empty_like(neighbor_positions)
    for batch in range(pos.size(0)):
        expected_positions[batch] = pos[batch, indices[batch]]
    torch.testing.assert_close(neighbor_positions, expected_positions)

    actual_distances = (neighbor_positions - q[:, :, :, None, :]).square().sum(dim=-1)
    torch.testing.assert_close(distances, actual_distances, rtol=1.0e-5, atol=1.0e-5)


def test_query_displaced_knn_matches_flattened_query_knn_batched():
    assert torch.cuda.is_available()
    pos, q, _ = _displaced_query_inputs(batch_size=2, n=32, heads=4, dim=3)
    batch_size, n, heads, dim = q.shape
    flat_queries = q.reshape(batch_size, n * heads, dim).contiguous()

    bvh = torchbvh.build_bvh_batched(pos)
    flat_indices, flat_distances, flat_positions = torchbvh.query_knn_batched(
        bvh,
        flat_queries,
        8,
        source_points=pos,
    )
    indices, distances, neighbor_positions = torchbvh.query_displaced_knn(pos, q, 8)

    torch.testing.assert_close(indices, flat_indices.reshape(batch_size, n, heads, 8))
    torch.testing.assert_close(distances, flat_distances.reshape(batch_size, n, heads, 8))
    torch.testing.assert_close(
        neighbor_positions,
        flat_positions.reshape(batch_size, n, heads, 8, dim),
    )


def test_query_displaced_knn_builds_one_bvh_per_sample_and_flattens_only_queries(monkeypatch):
    assert torch.cuda.is_available()
    pos, q, _ = _displaced_query_inputs(batch_size=2, n=10, heads=3, dim=3)
    calls = []

    def fake_build_bvh_batched(points_arg):
        calls.append(("build", points_arg.shape, points_arg.requires_grad))
        return {"_batched": True}

    def fake_query_knn_batched(bvh, query_points, k, *, source_points=None, sort_queries=True):
        calls.append(
            (
                "query",
                query_points.shape,
                k,
                None if source_points is None else source_points.shape,
                sort_queries,
            )
        )
        batch_size, query_count, dim = query_points.shape
        indices = torch.zeros((batch_size, query_count, k), device=query_points.device, dtype=torch.int64)
        distances = torch.zeros((batch_size, query_count, k), device=query_points.device, dtype=torch.float32)
        if source_points is None:
            return indices, distances
        positions = torch.zeros((batch_size, query_count, k, dim), device=query_points.device, dtype=torch.float32)
        return indices, distances, positions

    monkeypatch.setattr(multihead_module, "build_bvh_batched", fake_build_bvh_batched)
    monkeypatch.setattr(multihead_module, "query_knn_batched", fake_query_knn_batched)

    indices, distances, positions = torchbvh.query_displaced_knn(pos.requires_grad_(), q.requires_grad_(), 4)

    assert indices.shape == (2, 10, 3, 4)
    assert distances.shape == (2, 10, 3, 4)
    assert positions.shape == (2, 10, 3, 4, 3)
    assert [call[0] for call in calls] == ["build", "query"]
    assert calls[0] == ("build", pos.shape, False)
    assert calls[1] == ("query", (2, 30, 3), 4, pos.shape, True)


def test_query_displaced_knn_can_skip_position_gather():
    assert torch.cuda.is_available()
    pos, q, _ = _displaced_query_inputs(batch_size=2, n=20, heads=2, dim=2)

    result = torchbvh.query_displaced_knn(pos, q, 4, return_positions=False)

    assert len(result) == 2
    indices, distances = result
    assert indices.shape == (2, 20, 2, 4)
    assert distances.shape == (2, 20, 2, 4)


def test_gather_neighbor_values_uses_matching_head_only():
    assert torch.cuda.is_available()
    values = torch.arange(2 * 6 * 3 * 4, device="cuda", dtype=torch.float32).reshape(2, 6, 3, 4)
    indices = torch.tensor(
        [
            [
                [[0, 5], [1, 4], [2, 3]],
                [[3, 2], [4, 1], [5, 0]],
                [[1, 0], [2, 5], [3, 4]],
                [[4, 3], [5, 2], [0, 1]],
                [[2, 1], [3, 0], [4, 5]],
                [[5, 4], [0, 3], [1, 2]],
            ],
            [
                [[5, 0], [4, 1], [3, 2]],
                [[2, 3], [1, 4], [0, 5]],
                [[0, 1], [5, 2], [4, 3]],
                [[3, 4], [2, 5], [1, 0]],
                [[1, 2], [0, 3], [5, 4]],
                [[4, 5], [3, 0], [2, 1]],
            ],
        ],
        device="cuda",
        dtype=torch.int64,
    ).contiguous()

    gathered = torchbvh.gather_neighbor_values(values.contiguous(), indices)

    assert gathered.shape == (2, 6, 3, 2, 4)
    expected = torch.empty_like(gathered)
    for batch in range(2):
        for point in range(6):
            for head in range(3):
                expected[batch, point, head] = values[batch, indices[batch, point, head], head]
    torch.testing.assert_close(gathered, expected)


def test_query_displaced_knn_rejects_unsupported_k_like_batched_query():
    assert torch.cuda.is_available()
    pos, q, _ = _displaced_query_inputs(batch_size=2, n=16, heads=2, dim=2)

    with pytest.raises(ValueError, match="k must be 4, 8, or 16"):
        torchbvh.query_displaced_knn(pos, q, 5)


def test_interpolate_displaced_weighted_mean_matches_reference():
    assert torch.cuda.is_available()
    pos, q, values = _displaced_query_inputs(batch_size=2, n=24, heads=2, dim=3, channels=3)

    output = torchbvh.interpolate_displaced(pos, q, values, 4)
    indices, squared_distances = torchbvh.query_displaced_knn(
        pos,
        q,
        4,
        return_positions=False,
    )
    assert not (squared_distances <= 1.0e-12).any()
    expected = _old_style_weighted_mean_reference(values, indices, squared_distances)

    assert output.shape == (2, 24, 2, 3)
    torch.testing.assert_close(output, expected, rtol=1.0e-5, atol=1.0e-5)


def test_interpolate_displaced_exact_hits_match_old_style_reference():
    assert torch.cuda.is_available()
    pos, q, values = _robust_displaced_query_inputs(dim=2, k=4)

    output = torchbvh.interpolate_displaced(pos, q, values, 4)
    indices, squared_distances = torchbvh.query_displaced_knn(
        pos,
        q,
        4,
        return_positions=False,
    )
    exact_mask = squared_distances <= 1.0e-12
    assert exact_mask[:, 0, 0].all()
    expected = _old_style_weighted_mean_reference(values, indices, squared_distances)

    assert output.shape == (2, 32, 3, 5)
    torch.testing.assert_close(output, expected, rtol=1.0e-5, atol=1.0e-5)


@pytest.mark.parametrize(("dim", "k"), [(2, 4), (3, 8), (3, 16)])
def test_displaced_robust_geometry_matches_flattened_batched_oracles(dim, k):
    assert torch.cuda.is_available()
    pos, q, values = _robust_displaced_query_inputs(dim=dim, k=k)
    batch_size, n, heads, _ = q.shape

    bvh = torchbvh.build_bvh_batched(pos)
    flat_q = q.reshape(batch_size, n * heads, dim).contiguous()
    flat_indices, flat_distances, flat_positions = torchbvh.query_knn_batched(
        bvh,
        flat_q,
        k,
        source_points=pos,
    )
    indices, distances, neighbor_positions = torchbvh.query_displaced_knn(pos, q, k)

    assert int(indices.min()) >= 0
    assert int(indices.max()) < pos.size(1)
    assert torch.all(distances[:, :, :, :-1] <= distances[:, :, :, 1:])
    torch.testing.assert_close(indices, flat_indices.reshape(batch_size, n, heads, k))
    torch.testing.assert_close(distances, flat_distances.reshape(batch_size, n, heads, k))
    torch.testing.assert_close(neighbor_positions, flat_positions.reshape(batch_size, n, heads, k, dim))

    gathered = torchbvh.gather_neighbor_values(values, indices)
    expected_gathered = torch.empty_like(gathered)
    for batch in range(batch_size):
        for point in range(n):
            for head in range(heads):
                expected_gathered[batch, point, head] = values[batch, indices[batch, point, head], head]
    torch.testing.assert_close(gathered, expected_gathered)


def test_interpolate_displaced_exact_hit_duplicate_weighted_mean_and_gradient_boundary():
    assert torch.cuda.is_available()
    pos, q, values = _robust_displaced_query_inputs(dim=2, k=4)
    pos = pos.detach().requires_grad_(True)
    q = q.detach().requires_grad_(True)
    values = values.detach().requires_grad_(True)

    output = torchbvh.interpolate_displaced(pos, q, values, 4)
    indices, squared_distances = torchbvh.query_displaced_knn(
        pos,
        q,
        4,
        return_positions=False,
    )
    neighbor_values = torchbvh.gather_neighbor_values(values, indices)
    exact_mask = squared_distances <= 1.0e-12
    assert exact_mask[:, 0, 0].all()

    exact_weights = exact_mask.to(neighbor_values.dtype).unsqueeze(-1)
    exact_counts = exact_weights.sum(dim=-2).clamp(min=1.0)
    expected_exact_average = (neighbor_values * exact_weights).sum(dim=-2) / exact_counts
    torch.testing.assert_close(output[exact_mask.any(dim=-1)], expected_exact_average[exact_mask.any(dim=-1)])

    output.square().sum().backward()
    assert values.grad is not None
    assert torch.count_nonzero(values.grad) > 0
    assert pos.grad is None
    assert q.grad is None


def test_gather_neighbor_values_rejects_invalid_values_dtype_and_device():
    assert torch.cuda.is_available()
    values = torch.randn((2, 6, 3, 4), device="cuda", dtype=torch.float32).contiguous()
    indices = torch.zeros((2, 6, 3, 2), device="cuda", dtype=torch.int64).contiguous()

    with pytest.raises(ValueError, match="values must be float32"):
        torchbvh.gather_neighbor_values(values.double().contiguous(), indices)
    with pytest.raises(ValueError, match="values must be a CUDA tensor"):
        torchbvh.gather_neighbor_values(values.cpu().contiguous(), indices.cpu())


def test_interpolate_displaced_rejects_unsupported_reduction():
    assert torch.cuda.is_available()
    pos, q, values = _displaced_query_inputs(batch_size=2, n=16, heads=2, dim=2)

    with pytest.raises(ValueError, match="reduction must be 'weighted_mean'"):
        torchbvh.interpolate_displaced(pos, q, values, 4, reduction="mean")


def test_query_displaced_knn_gradient_boundary_outputs_are_detached():
    assert torch.cuda.is_available()
    pos, q, _ = _displaced_query_inputs(batch_size=2, n=20, heads=2, dim=3)
    pos = pos.detach().requires_grad_(True)
    q = q.detach().requires_grad_(True)

    indices, distances, neighbor_positions = torchbvh.query_displaced_knn(pos, q, 4)

    assert not indices.requires_grad
    assert not distances.requires_grad
    assert not neighbor_positions.requires_grad
    assert pos.grad is None
    assert q.grad is None


def test_gather_neighbor_values_gradients_flow_to_values_only():
    assert torch.cuda.is_available()
    values = torch.randn((2, 6, 3, 4), device="cuda", dtype=torch.float32).contiguous().requires_grad_()
    indices = torch.tensor(
        [
            [
                [[0, 1], [1, 2], [2, 3]],
                [[3, 4], [4, 5], [5, 0]],
                [[1, 2], [2, 3], [3, 4]],
                [[4, 5], [5, 0], [0, 1]],
                [[2, 3], [3, 4], [4, 5]],
                [[5, 0], [0, 1], [1, 2]],
            ],
            [
                [[5, 4], [4, 3], [3, 2]],
                [[2, 1], [1, 0], [0, 5]],
                [[0, 5], [5, 4], [4, 3]],
                [[3, 2], [2, 1], [1, 0]],
                [[1, 0], [0, 5], [5, 4]],
                [[4, 3], [3, 2], [2, 1]],
            ],
        ],
        device="cuda",
        dtype=torch.int64,
    ).contiguous()

    gathered = torchbvh.gather_neighbor_values(values, indices)
    gathered.square().sum().backward()

    assert values.grad is not None
    assert torch.count_nonzero(values.grad) > 0
