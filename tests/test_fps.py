import math

import pytest
import torch

import torchbvh
import torchbvh._fps as fps_module
from torchbvh._fps import (
    _fps_approx_bucketed,
    _fps_exact_bucketed,
    _fps_exact_full_scan,
)


def _independent_fps(points: torch.Tensor, M: int, seed: int = 0):
    if seed == -1:
        center = 0.5 * (points.min(dim=0).values + points.max(dim=0).values)
        seed = int((points - center).square().sum(dim=1).argmin().item())
    fps_idx = torch.empty(M, device=points.device, dtype=torch.int64)
    fps_idx[0] = seed
    dist = (points - points[seed]).square().sum(dim=1)
    nearest_anchor = torch.zeros(points.size(0), device=points.device, dtype=torch.int32)
    for round_idx in range(1, M):
        idx = int(torch.argmax(dist).item())
        fps_idx[round_idx] = idx
        d_new = (points - points[idx]).square().sum(dim=1)
        update = d_new < dist
        dist = torch.where(update, d_new, dist)
        nearest_anchor = torch.where(
            update,
            torch.full_like(nearest_anchor, round_idx),
            nearest_anchor,
        )
    return fps_idx, nearest_anchor, dist


def _exact_reference(points: torch.Tensor, M: int, seed: int = 0):
    fps_idx, nearest_anchor, nearest_dist = _independent_fps(points, M, seed=seed)
    anchor_counts = torch.bincount(nearest_anchor.to(torch.int64), minlength=M).to(torch.int32)
    anchor_radius = torch.zeros(M, device=points.device, dtype=points.dtype)
    for anchor in range(M):
        mask = nearest_anchor == anchor
        if bool(mask.any()):
            anchor_radius[anchor] = nearest_dist[mask].max()

    bvh = torchbvh.build_bvh(points)
    sorted_indices = bvh["sorted_indices"]
    reverse_leaf_pos = torch.empty(points.size(0), device=points.device, dtype=torch.int64)
    reverse_leaf_pos[sorted_indices] = torch.arange(points.size(0), device=points.device)
    coarse_order = torch.argsort(reverse_leaf_pos[fps_idx], stable=True)

    return torchbvh.FPSResult(
        indices=fps_idx,
        points=points[fps_idx],
        nearest_anchor=nearest_anchor,
        nearest_anchor_dist_sq=nearest_dist,
        anchor_radius=anchor_radius,
        anchor_counts=anchor_counts,
        coarse_order=coarse_order,
        selection_order_indices=fps_idx,
    )


def _leaf_level(t: int) -> int:
    return 0 if t <= 1 else math.ceil(math.log2(t))


def _virtual_leaves(t: int) -> int:
    return (1 << _leaf_level(t)) - t


def _virtual_nodes_before_level(t: int, level: int) -> int:
    lvl = _virtual_leaves(t) >> (_leaf_level(t) - level + 1)
    return 2 * lvl - lvl.bit_count()


def _real_nodes_at_level(t: int, level: int) -> int:
    return (1 << level) - (_virtual_leaves(t) >> (_leaf_level(t) - level))


def _first_index_at_level(level: int) -> int:
    return (1 << level) - 1


def _level_of(implicit_idx: int) -> int:
    return (implicit_idx + 1).bit_length() - 1


def _memory_index(t: int, implicit_idx: int) -> int:
    return implicit_idx - _virtual_nodes_before_level(t, _level_of(implicit_idx))


def _is_virtual(t: int, implicit_idx: int) -> bool:
    level = _level_of(implicit_idx)
    return implicit_idx - _first_index_at_level(level) >= _real_nodes_at_level(t, level)


def _assert_result_invariants(result, points: torch.Tensor, M: int):
    N, D = points.shape
    assert isinstance(result, torchbvh.FPSResult)
    assert result.indices.shape == (M,)
    assert result.points.shape == (M, D)
    assert result.nearest_anchor.shape == (N,)
    assert result.nearest_anchor_dist_sq.shape == (N,)
    assert result.anchor_radius.shape == (M,)
    assert result.anchor_counts.shape == (M,)
    assert result.coarse_order.shape == (M,)
    assert result.selection_order_indices.shape == (M,)

    for tensor in (
        result.indices,
        result.points,
        result.nearest_anchor,
        result.nearest_anchor_dist_sq,
        result.anchor_radius,
        result.anchor_counts,
        result.coarse_order,
        result.selection_order_indices,
    ):
        assert tensor.is_cuda
    assert result.indices.dtype == torch.int64
    assert result.selection_order_indices.dtype == torch.int64
    assert result.nearest_anchor.dtype == torch.int32
    assert result.nearest_anchor_dist_sq.dtype == torch.float32
    assert result.anchor_radius.dtype == torch.float32
    assert result.anchor_counts.dtype == torch.int32
    assert torch.isfinite(result.points).all()
    assert torch.isfinite(result.nearest_anchor_dist_sq).all()
    assert torch.isfinite(result.anchor_radius).all()

    assert torch.equal(result.selection_order_indices, result.indices)
    assert torch.equal(result.points, points[result.indices])
    assert torch.unique(result.indices).numel() == M
    assert torch.all(result.indices >= 0)
    assert torch.all(result.indices < N)
    assert torch.all(result.nearest_anchor >= 0)
    assert torch.all(result.nearest_anchor < M)
    assert int(result.anchor_counts.sum().item()) == N
    assert torch.equal(torch.sort(result.coarse_order).values, torch.arange(M, device=points.device))


def _assert_batched_result_invariants(result, points: torch.Tensor, M: int):
    B, N, D = points.shape
    assert isinstance(result, torchbvh.FPSResult)
    assert result.indices.shape == (B, M)
    assert result.points.shape == (B, M, D)
    assert result.nearest_anchor.shape == (B, N)
    assert result.nearest_anchor_dist_sq.shape == (B, N)
    assert result.anchor_radius.shape == (B, M)
    assert result.anchor_counts.shape == (B, M)
    assert result.coarse_order.shape == (B, M)
    assert result.selection_order_indices.shape == (B, M)
    for b in range(B):
        single = torchbvh.FPSResult(
            indices=result.indices[b],
            points=result.points[b],
            nearest_anchor=result.nearest_anchor[b],
            nearest_anchor_dist_sq=result.nearest_anchor_dist_sq[b],
            anchor_radius=result.anchor_radius[b],
            anchor_counts=result.anchor_counts[b],
            coarse_order=result.coarse_order[b],
            selection_order_indices=result.selection_order_indices[b],
        )
        _assert_result_invariants(single, points[b], M)


def _assert_result_metadata_matches_manual(result, points: torch.Tensor):
    batched_points = points if points.dim() == 3 else points.unsqueeze(0)
    batched_result = result
    if points.dim() == 2:
        batched_result = torchbvh.FPSResult(
            indices=result.indices.unsqueeze(0),
            points=result.points.unsqueeze(0),
            nearest_anchor=result.nearest_anchor.unsqueeze(0),
            nearest_anchor_dist_sq=result.nearest_anchor_dist_sq.unsqueeze(0),
            anchor_radius=result.anchor_radius.unsqueeze(0),
            anchor_counts=result.anchor_counts.unsqueeze(0),
            coarse_order=result.coarse_order.unsqueeze(0),
            selection_order_indices=result.selection_order_indices.unsqueeze(0),
        )

    B, N, _D = batched_points.shape
    M = batched_result.indices.size(1)
    expected_counts = torch.zeros((B, M), device=batched_points.device, dtype=torch.int32)
    expected_counts.scatter_add_(
        1,
        batched_result.nearest_anchor.long(),
        torch.ones((B, N), device=batched_points.device, dtype=torch.int32),
    )
    expected_radius = torch.zeros((B, M), device=batched_points.device, dtype=torch.float32)
    expected_radius.scatter_reduce_(
        1,
        batched_result.nearest_anchor.long(),
        batched_result.nearest_anchor_dist_sq,
        reduce="amax",
        include_self=True,
    )
    torch.testing.assert_close(batched_result.anchor_counts, expected_counts)
    torch.testing.assert_close(batched_result.anchor_radius, expected_radius)
    torch.testing.assert_close(
        batched_result.anchor_counts.sum(dim=1),
        torch.full((B,), N, device=batched_points.device, dtype=torch.int64),
    )


def _expected_right_child_mem(N: int) -> torch.Tensor:
    leaf_level = _leaf_level(N)
    values = [0] * (2 * N - 1 + ((1 << leaf_level) - N).bit_count())
    for implicit_idx in range((1 << (leaf_level + 1)) - 1):
        if _is_virtual(N, implicit_idx):
            continue
        mem_idx = _memory_index(N, implicit_idx)
        if _level_of(implicit_idx) == leaf_level:
            values[mem_idx] = -1
            continue
        right = 2 * implicit_idx + 2
        values[mem_idx] = -1 if _is_virtual(N, right) else _memory_index(N, right)
    return torch.tensor(values, dtype=torch.int32)


# Public API behavior and exact correctness.


@pytest.mark.parametrize("N", [3, 5, 7, 8, 15, 16, 17, 50000])
def test_bvh_right_child_mem_matches_implicit_tree(N):
    assert torch.cuda.is_available()
    torch.manual_seed(12900 + N)
    points = torch.rand((N, 3), device="cuda", dtype=torch.float32).contiguous()
    single = torchbvh.build_bvh(points)
    batched_points = torch.stack((points, torch.flip(points, dims=(0,)).contiguous()), dim=0).contiguous()
    batched = torchbvh.build_bvh_batched(batched_points)

    expected = _expected_right_child_mem(N)
    torch.testing.assert_close(single["right_child_mem"].cpu(), expected)
    torch.testing.assert_close(batched["right_child_mem"].cpu(), expected)

    leaf_level = _leaf_level(N)
    left_child = single["left_child_mem"].cpu()
    mem_to_leaf = single["mem_to_leaf"].cpu()
    for implicit_idx in range((1 << (leaf_level + 1)) - 1):
        if _is_virtual(N, implicit_idx):
            continue
        mem_idx = _memory_index(N, implicit_idx)
        is_leaf = _level_of(implicit_idx) == leaf_level
        if is_leaf:
            assert int(single["right_child_mem"][mem_idx].item()) == -1
            assert int(left_child[mem_idx].item()) == -1
            assert int(mem_to_leaf[mem_idx].item()) == implicit_idx - _first_index_at_level(leaf_level)
        else:
            assert int(mem_to_leaf[mem_idx].item()) == -1
            assert int(left_child[mem_idx].item()) == _memory_index(N, 2 * implicit_idx + 1)


@pytest.mark.parametrize("N", [16, 1024])
@pytest.mark.parametrize("D", [2, 3])
def test_exact_reference_matches_independent_brute_force(N, D):
    assert torch.cuda.is_available()
    torch.manual_seed(12000 + N + D)
    points = torch.rand((N, D), device="cuda", dtype=torch.float32).contiguous()
    for M in (4, N // 4, N // 2):
        result = _exact_reference(points, M, seed=0)
        expected_idx, expected_nearest, expected_dist = _independent_fps(points, M, seed=0)
        _assert_result_invariants(result, points, M)
        torch.testing.assert_close(result.indices, expected_idx)
        torch.testing.assert_close(result.nearest_anchor, expected_nearest)
        torch.testing.assert_close(result.nearest_anchor_dist_sq, expected_dist)


@pytest.mark.parametrize("N", [16, 1024])
@pytest.mark.parametrize("D", [2, 3])
def test_cuda_full_scan_matches_exact_reference_seed_zero(N, D):
    assert torch.cuda.is_available()
    torch.manual_seed(12600 + N + D)
    points = torch.rand((N, D), device="cuda", dtype=torch.float32).contiguous()
    for M in (4, N // 4, N // 2):
        result = _fps_exact_full_scan(points, M, seed=0)
        expected = _exact_reference(points, M, seed=0)
        _assert_result_invariants(result, points, M)
        for field in result.__dataclass_fields__:
            torch.testing.assert_close(getattr(result, field), getattr(expected, field))


@pytest.mark.parametrize("D", [2, 3])
def test_cuda_full_scan_matches_exact_reference_aabb_center_seed(D):
    assert torch.cuda.is_available()
    torch.manual_seed(12700 + D)
    points = torch.rand((257, D), device="cuda", dtype=torch.float32).contiguous()
    result = _fps_exact_full_scan(points, 64, seed=-1)
    expected = _exact_reference(points, 64, seed=-1)
    _assert_result_invariants(result, points, 64)
    for field in result.__dataclass_fields__:
        torch.testing.assert_close(getattr(result, field), getattr(expected, field))


def test_exact_reference_is_deterministic_for_fixed_seed():
    assert torch.cuda.is_available()
    torch.manual_seed(12100)
    points = torch.rand((257, 3), device="cuda", dtype=torch.float32).contiguous()
    first = torchbvh.fps(points, 64, seed=7, mode="exact_bucketed")
    second = torchbvh.fps(points, 64, seed=7, mode="exact_bucketed")
    for field in first.__dataclass_fields__:
        torch.testing.assert_close(getattr(first, field), getattr(second, field))


@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("N", [16, 1024])
@pytest.mark.parametrize("D", [2, 3])
@pytest.mark.parametrize("seed", [0, -1])
def test_cuda_full_scan_batched_metadata_matches_exact_reference(B, N, D, seed):
    assert torch.cuda.is_available()
    torch.manual_seed(12800 + B + N + D + seed)
    points = torch.rand((B, N, D), device="cuda", dtype=torch.float32).contiguous()
    M = 4 if N == 16 else 128
    result = torchbvh.fps(points, M, seed=seed, mode="exact_bucketed")
    _assert_batched_result_invariants(result, points, M)
    for b in range(B):
        expected = _exact_reference(points[b].contiguous(), M, seed=seed)
        for field in result.__dataclass_fields__:
            torch.testing.assert_close(getattr(result, field)[b], getattr(expected, field))


def test_public_exact_fps_preserves_exact_semantics():
    assert torch.cuda.is_available()
    torch.manual_seed(13400)
    points = torch.rand((4, 128, 3), device="cuda", dtype=torch.float32).contiguous()
    result = torchbvh.fps(points, 32, seed=-1, mode="exact_bucketed")
    _assert_batched_result_invariants(result, points, 32)
    for b in range(points.size(0)):
        expected = _exact_reference(points[b].contiguous(), 32, seed=-1)
        for field in result.__dataclass_fields__:
            torch.testing.assert_close(getattr(result, field)[b], getattr(expected, field))


def test_public_default_exact_bucketed_matches_full_scan_reference():
    assert torch.cuda.is_available()
    for N in (1024, 4096):
        torch.manual_seed(15200 + N)
        points = torch.rand((2, N, 3), device="cuda", dtype=torch.float32).contiguous()
        M = 32 if N == 1024 else 64
        result = torchbvh.fps(points, M, seed=0, mode="exact_bucketed")
        expected = _fps_exact_full_scan(points, M, seed=0)
        _assert_batched_result_invariants(result, points, M)
        for field in result.__dataclass_fields__:
            torch.testing.assert_close(getattr(result, field), getattr(expected, field))


def test_public_exact_full_scan_mode_matches_full_scan_reference():
    assert torch.cuda.is_available()
    for N in (1024, 4096):
        torch.manual_seed(15240 + N)
        points = torch.rand((2, N, 3), device="cuda", dtype=torch.float32).contiguous()
        M = 32 if N == 1024 else 64
        result = torchbvh.fps(
            points,
            M,
            seed=0,
            mode="exact_full_scan",
        )
        expected = _fps_exact_full_scan(points, M, seed=0)
        _assert_batched_result_invariants(result, points, M)
        for field in result.__dataclass_fields__:
            torch.testing.assert_close(getattr(result, field), getattr(expected, field))


def test_anchor_radius_and_counts_match_manual_loop_small_case():
    assert torch.cuda.is_available()
    torch.manual_seed(12200)
    points = torch.rand((64, 2), device="cuda", dtype=torch.float32).contiguous()
    result = _exact_reference(points, 16, seed=3)
    for anchor in range(result.indices.numel()):
        mask = result.nearest_anchor == anchor
        assert int(result.anchor_counts[anchor].item()) == int(mask.sum().item())
        if bool(mask.any()):
            torch.testing.assert_close(
                result.anchor_radius[anchor],
                result.nearest_anchor_dist_sq[mask].max(),
            )
        else:
            assert result.anchor_radius[anchor].item() == 0.0


def test_coarse_order_gathers_morton_ordered_anchor_points():
    assert torch.cuda.is_available()
    torch.manual_seed(12400)
    points = torch.rand((128, 3), device="cuda", dtype=torch.float32).contiguous()
    result = _exact_reference(points, 32, seed=0)
    morton_points = points[result.indices[result.coarse_order]]
    torch.testing.assert_close(morton_points, result.points[result.coarse_order])


def test_nearest_anchor_remapping_into_morton_ordered_anchor_tensor():
    assert torch.cuda.is_available()
    points = torch.tensor(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [0.0, 4.0],
            [4.0, 4.0],
            [2.0, 2.0],
            [2.1, 2.0],
            [1.9, 2.0],
            [2.0, 2.1],
        ],
        device="cuda",
        dtype=torch.float32,
    ).contiguous()
    result = _exact_reference(points, 4, seed=0)
    morton_anchor_points = points[result.indices[result.coarse_order]]
    selection_to_morton = torch.empty_like(result.coarse_order)
    selection_to_morton[result.coarse_order] = torch.arange(
        result.coarse_order.numel(),
        device=points.device,
        dtype=result.coarse_order.dtype,
    )
    remapped_points = morton_anchor_points[selection_to_morton[result.nearest_anchor.long()]]
    torch.testing.assert_close(remapped_points, result.points[result.nearest_anchor.long()])


def test_fps_accepts_batched_inputs_on_cuda_path():
    assert torch.cuda.is_available()
    points = torch.rand((2, 16, 3), device="cuda", dtype=torch.float32).contiguous()
    result = torchbvh.fps(points, 4, mode="exact_bucketed")
    _assert_batched_result_invariants(result, points, 4)
    default_result = torchbvh.fps(points, 4)
    _assert_batched_result_invariants(default_result, points, 4)


def test_public_fps_default_routes_to_exact_bucketed_graph(monkeypatch):
    assert torch.cuda.is_available()
    points = torch.rand((2, 16, 3), device="cuda", dtype=torch.float32).contiguous()
    sentinel = torchbvh.FPSResult(
        indices=torch.empty((2, 4), device="cuda", dtype=torch.int64),
        points=torch.empty((2, 4, 3), device="cuda", dtype=torch.float32),
        nearest_anchor=torch.empty((2, 16), device="cuda", dtype=torch.int32),
        nearest_anchor_dist_sq=torch.empty((2, 16), device="cuda", dtype=torch.float32),
        anchor_radius=torch.empty((2, 4), device="cuda", dtype=torch.float32),
        anchor_counts=torch.empty((2, 4), device="cuda", dtype=torch.int32),
        coarse_order=torch.empty((2, 4), device="cuda", dtype=torch.int64),
        selection_order_indices=torch.empty((2, 4), device="cuda", dtype=torch.int64),
    )
    calls = []

    def fake_exact_bucketed(points_arg, target_tokens, **kwargs):
        calls.append((points_arg, target_tokens, kwargs))
        return sentinel

    monkeypatch.setattr(fps_module, "_fps_exact_bucketed", fake_exact_bucketed)

    result = torchbvh.fps(points, 4)

    assert result is sentinel
    assert len(calls) == 1
    points_arg, target_tokens, kwargs = calls[0]
    assert points_arg is points
    assert target_tokens == 4
    assert kwargs["seed"] == 0
    assert kwargs["bucket_size"] == 256
    assert kwargs["use_graph"] is True
    assert kwargs["enable_pruning"] is True
    assert kwargs["return_diagnostics"] is False


def test_public_fps_single_route_is_exact_and_ignores_experimental_knobs():
    assert torch.cuda.is_available()
    torch.manual_seed(15300)
    points = torch.rand((256, 3), device="cuda", dtype=torch.float32).contiguous()
    exact = torchbvh.fps(points, 64, seed=0, mode="exact_bucketed")
    default = torchbvh.fps(points, 64, seed=0)
    full_scan = torchbvh.fps(points, 64, seed=0, mode="exact_full_scan")
    bucketed = torchbvh.fps(points, 64, seed=0, mode="exact_bucketed")
    expected = _fps_exact_full_scan(points, 64, seed=0)

    for result in (default, exact, full_scan, bucketed):
        _assert_result_invariants(result, points, 64)
        for field in result.__dataclass_fields__:
            torch.testing.assert_close(getattr(result, field), getattr(expected, field))


def test_public_fps_approx_bucketed_mode_is_opt_in_approximate_route():
    assert torch.cuda.is_available()
    torch.manual_seed(15340)
    points = torch.rand((2, 512, 3), device="cuda", dtype=torch.float32).contiguous()
    approx = torchbvh.fps(
        points,
        128,
        seed=0,
        mode="approx_bucketed",
        r=4,
        c=2,
        alpha=0.25,
    )
    _assert_batched_result_invariants(approx, points, 128)
    assert min(torch.unique(approx.indices[b]).numel() for b in range(points.size(0))) == 128


def test_public_fps_rejects_invalid_modes_and_old_public_name_is_absent():
    assert torch.cuda.is_available()
    points = torch.rand((64, 3), device="cuda", dtype=torch.float32).contiguous()
    with pytest.raises(ValueError, match="mode must be one of"):
        torchbvh.fps(points, 16, mode="bogus")
    assert hasattr(torchbvh, "fps")
    assert not hasattr(torchbvh, "fps_" + "downsample_geometry")


def test_public_fps_approximate_route_validates_approximate_knobs():
    assert torch.cuda.is_available()
    points = torch.rand((64, 3), device="cuda", dtype=torch.float32).contiguous()
    with pytest.raises(ValueError, match="r must be"):
        torchbvh.fps(points, 16, mode="approx_bucketed", r=0)
    with pytest.raises(ValueError, match="c must be"):
        torchbvh.fps(points, 16, mode="approx_bucketed", c=0)
    with pytest.raises(ValueError, match=r"r \* c"):
        torchbvh.fps(points, 16, mode="approx_bucketed", r=8, c=5)


# Approximate-mode invariants.


def test_bvh_bucket_queue_approx_nearest_assignment_exact_for_selected_anchors():
    assert torch.cuda.is_available()
    torch.manual_seed(15440)
    points = torch.rand((192, 3), device="cuda", dtype=torch.float32).contiguous()
    result = _fps_approx_bucketed(
        points,
        48,
        seed=0,
        bucket_size=32,
        refresh_interval=2,
        candidates_per_round=8,
        anchors_per_round=4,
        alpha=0.0,
    )
    _assert_result_invariants(result, points, 48)
    distances = torch.cdist(points, result.points).square()
    expected_dist, expected_anchor = distances.min(dim=1)
    torch.testing.assert_close(result.nearest_anchor_dist_sq, expected_dist, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(result.nearest_anchor.to(torch.int64), expected_anchor)


def test_bvh_bucket_queue_approx_quality_smoke_against_exact():
    assert torch.cuda.is_available()
    torch.manual_seed(15450)
    points = torch.rand((512, 3), device="cuda", dtype=torch.float32).contiguous()
    exact = torchbvh.fps(points, 128, seed=0, mode="exact_bucketed")
    approx = _fps_approx_bucketed(
        points,
        128,
        seed=0,
        bucket_size=64,
        refresh_interval=2,
        candidates_per_round=8,
        anchors_per_round=4,
        alpha=0.0,
    )
    _assert_result_invariants(approx, points, 128)
    mean_ratio = approx.nearest_anchor_dist_sq.clamp_min(0).sqrt().mean() / exact.nearest_anchor_dist_sq.clamp_min(0).sqrt().mean()
    max_ratio = approx.nearest_anchor_dist_sq.max().sqrt() / exact.nearest_anchor_dist_sq.max().sqrt()
    assert mean_ratio < 1.25
    assert max_ratio < 1.35


def test_bvh_bucket_queue_approx_handles_more_anchors_than_buckets():
    assert torch.cuda.is_available()
    torch.manual_seed(15460)
    points = torch.rand((2, 1024, 3), device="cuda", dtype=torch.float32).contiguous()
    result = _fps_approx_bucketed(
        points,
        256,
        seed=0,
        bucket_size=256,
        refresh_interval=8,
        candidates_per_round=32,
        anchors_per_round=8,
        alpha=0.0,
    )
    _assert_batched_result_invariants(result, points, 256)
    assert min(torch.unique(result.indices[b]).numel() for b in range(points.size(0))) == 256


# Exact-bucketed graph/cache behavior.


@pytest.mark.parametrize(
    ("B", "N", "D", "M", "seed", "bucket_size"),
    [
        (1, 16, 2, 4, 0, 8),
        (4, 128, 3, 32, -1, 32),
        (1, 1024, 3, 256, 0, 128),
        (1, 4096, 3, 1024, 0, 256),
    ],
)
def test_exact_bucketed_matches_public_exact(B, N, D, M, seed, bucket_size):
    assert torch.cuda.is_available()
    torch.manual_seed(15500 + N + D + B)
    points = torch.rand((B, N, D), device="cuda", dtype=torch.float32).contiguous()
    exact = torchbvh.fps(points, M, seed=seed, mode="exact_bucketed")
    bucketed, diagnostics = _fps_exact_bucketed(
        points,
        M,
        seed=seed,
        bucket_size=bucket_size,
        enable_pruning=True,
        return_diagnostics=True,
    )
    _assert_batched_result_invariants(bucketed, points, M)
    torch.testing.assert_close(bucketed.indices, exact.indices)
    torch.testing.assert_close(bucketed.nearest_anchor, exact.nearest_anchor)
    torch.testing.assert_close(bucketed.nearest_anchor_dist_sq, exact.nearest_anchor_dist_sq)
    assert diagnostics["route"] == 1
    assert diagnostics["bucket_count"] >= 1
    assert diagnostics["bucket_size_requested"] == bucket_size


def test_exact_bucketed_pruning_toggle_preserves_exact_outputs():
    assert torch.cuda.is_available()
    torch.manual_seed(15580)
    centers = torch.rand((2, 8, 3), device="cuda", dtype=torch.float32)
    noise = 0.01 * torch.randn((2, 64, 8, 3), device="cuda", dtype=torch.float32)
    points = (centers[:, None, :, :] + noise).reshape(2, 512, 3).contiguous()
    pruned, pruned_diag = _fps_exact_bucketed(
        points,
        128,
        seed=0,
        bucket_size=64,
        enable_pruning=True,
        return_diagnostics=True,
    )
    unpruned, unpruned_diag = _fps_exact_bucketed(
        points,
        128,
        seed=0,
        bucket_size=64,
        enable_pruning=False,
        return_diagnostics=True,
    )
    torch.testing.assert_close(pruned.indices, unpruned.indices)
    torch.testing.assert_close(pruned.nearest_anchor, unpruned.nearest_anchor)
    torch.testing.assert_close(pruned.nearest_anchor_dist_sq, unpruned.nearest_anchor_dist_sq)
    assert unpruned_diag["total_skips"] == 0
    assert pruned_diag["total_refreshes"] + pruned_diag["total_skips"] == (
        points.size(0) * pruned_diag["bucket_count"] * 128
    )
    assert pruned_diag["total_skips"] > 0


def test_exact_bucketed_graph_and_nongraph_match_public_exact_and_do_not_leak_state():
    assert torch.cuda.is_available()
    torch.manual_seed(15610)
    points_a = torch.rand((2, 512, 3), device="cuda", dtype=torch.float32).contiguous()
    points_b = torch.rand((2, 512, 3), device="cuda", dtype=torch.float32).contiguous()
    exact_a = torchbvh.fps(points_a, 128, seed=0, mode="exact_bucketed")
    exact_b = torchbvh.fps(points_b, 128, seed=0, mode="exact_bucketed")

    graph_a, graph_diag = _fps_exact_bucketed(
        points_a,
        128,
        seed=0,
        bucket_size=64,
        use_graph=True,
        enable_pruning=True,
        return_diagnostics=True,
    )
    graph_b, graph_diag_b = _fps_exact_bucketed(
        points_b,
        128,
        seed=0,
        bucket_size=64,
        use_graph=True,
        enable_pruning=True,
        return_diagnostics=True,
    )
    nongraph_b, nongraph_diag = _fps_exact_bucketed(
        points_b,
        128,
        seed=0,
        bucket_size=64,
        use_graph=False,
        enable_pruning=True,
        return_diagnostics=True,
    )

    assert graph_diag["graph_captured"] is True
    assert graph_diag_b["graph_captured"] is True
    assert nongraph_diag["graph_captured"] is False
    torch.testing.assert_close(graph_a.indices, exact_a.indices)
    torch.testing.assert_close(graph_a.nearest_anchor, exact_a.nearest_anchor)
    torch.testing.assert_close(graph_b.indices, exact_b.indices)
    torch.testing.assert_close(graph_b.nearest_anchor, exact_b.nearest_anchor)
    torch.testing.assert_close(nongraph_b.indices, exact_b.indices)
    torch.testing.assert_close(nongraph_b.nearest_anchor, exact_b.nearest_anchor)


@pytest.mark.parametrize(
    ("B", "N", "D", "M", "bucket_size", "enable_pruning"),
    [
        (1, 128, 2, 32, 32, True),
        (2, 128, 3, 32, 32, True),
        (2, 256, 3, 32, 32, True),
        (2, 256, 3, 64, 32, True),
        (2, 256, 3, 64, 64, True),
        (2, 256, 3, 64, 64, False),
    ],
)
def test_exact_bucketed_graph_cache_key_variations(B, N, D, M, bucket_size, enable_pruning):
    assert torch.cuda.is_available()
    torch.manual_seed(15650 + B + N + D + M + bucket_size + int(enable_pruning))
    points = torch.rand((B, N, D), device="cuda", dtype=torch.float32).contiguous()
    exact = torchbvh.fps(points, M, seed=0, mode="exact_bucketed")
    bucketed, diagnostics = _fps_exact_bucketed(
        points,
        M,
        seed=0,
        bucket_size=bucket_size,
        use_graph=True,
        enable_pruning=enable_pruning,
        return_diagnostics=True,
    )
    torch.testing.assert_close(bucketed.indices, exact.indices)
    torch.testing.assert_close(bucketed.nearest_anchor, exact.nearest_anchor)
    assert diagnostics["graph_captured"] is True
    assert diagnostics["bucket_size_requested"] == bucket_size
    assert diagnostics["enable_pruning"] is enable_pruning


def test_public_approximate_quality_smoke_is_invariant_oriented():
    assert torch.cuda.is_available()
    torch.manual_seed(12500)
    points = torch.rand((1024, 3), device="cuda", dtype=torch.float32).contiguous()
    approx = torchbvh.fps(
        points,
        256,
        seed=0,
        mode="approx_bucketed",
        r=4,
        c=2,
        alpha=0.25,
    )
    _assert_result_invariants(approx, points, 256)
    _assert_result_metadata_matches_manual(approx, points)
    assert torch.isfinite(approx.nearest_anchor_dist_sq).all()
    assert float(approx.nearest_anchor_dist_sq.max().item()) > 0.0
