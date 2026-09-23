import pytest
import torch

import torchbvh


def _independent_fps(points: torch.Tensor, count: int, seed: int):
    if seed == -1:
        center = 0.5 * (points.min(0).values + points.max(0).values)
        seed = int((points - center).square().sum(-1).argmin())
    indices = torch.empty(count, device=points.device, dtype=torch.int64)
    indices[0] = seed
    nearest_distance = (points - points[seed]).square().sum(-1)
    nearest_anchor = torch.zeros(
        points.size(0), device=points.device, dtype=torch.int32
    )
    for anchor in range(1, count):
        selected = int(nearest_distance.argmax())
        indices[anchor] = selected
        candidate = (points - points[selected]).square().sum(-1)
        update = candidate < nearest_distance
        nearest_distance = torch.where(update, candidate, nearest_distance)
        nearest_anchor = torch.where(
            update, torch.full_like(nearest_anchor, anchor), nearest_anchor
        )
    return indices, nearest_anchor, nearest_distance


def _assert_result(result: torchbvh.FPSResult, points: torch.Tensor, count: int):
    batched = points.dim() == 3
    batch = points.size(0) if batched else None
    source_count = points.size(-2)
    dim = points.size(-1)
    prefix = (batch,) if batched else ()
    assert result.indices.shape == (*prefix, count)
    assert result.points.shape == (*prefix, count, dim)
    assert result.nearest_anchor.shape == (*prefix, source_count)
    assert result.nearest_anchor_dist_sq.shape == (*prefix, source_count)
    assert result.anchor_radius.shape == (*prefix, count)
    assert result.anchor_counts.shape == (*prefix, count)
    assert result.coarse_order.shape == (*prefix, count)
    assert result.selection_order_indices.shape == (*prefix, count)
    assert torch.equal(result.selection_order_indices, result.indices)

    batches = range(batch) if batched else (None,)
    for batch_index in batches:
        indices = result.indices if batch_index is None else result.indices[batch_index]
        sample = points if batch_index is None else points[batch_index]
        assert torch.unique(indices).numel() == count
        torch.testing.assert_close(
            result.points if batch_index is None else result.points[batch_index],
            sample[indices],
        )


def _assert_exact(result: torchbvh.FPSResult, points: torch.Tensor, count: int, seed: int):
    _assert_result(result, points, count)
    batches = range(points.size(0)) if points.dim() == 3 else (None,)
    for batch_index in batches:
        sample = points if batch_index is None else points[batch_index]
        expected = _independent_fps(sample, count, seed)
        actual = (
            result.indices,
            result.nearest_anchor,
            result.nearest_anchor_dist_sq,
        )
        if batch_index is not None:
            actual = tuple(value[batch_index] for value in actual)
        for actual_value, expected_value in zip(actual, expected):
            torch.testing.assert_close(actual_value, expected_value)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("seed", [0, -1, 7])
def test_exact_bucketed_matches_independent_fps(dim, seed):
    assert torch.cuda.is_available()
    torch.manual_seed(12000 + dim + seed)
    points = torch.rand((257, dim), device="cuda", dtype=torch.float32)
    result = torchbvh.fps(points, 64, seed=seed)
    _assert_exact(result, points, 64, seed)


@pytest.mark.parametrize("dim", [2, 3])
def test_exact_bucketed_batched_matches_independent_fps(dim):
    assert torch.cuda.is_available()
    torch.manual_seed(12100 + dim)
    points = torch.rand((3, 128, dim), device="cuda", dtype=torch.float32)
    result = torchbvh.fps(points, 32, seed=-1, mode="exact_bucketed")
    _assert_exact(result, points, 32, -1)


def test_exact_metadata_matches_selected_anchors():
    assert torch.cuda.is_available()
    torch.manual_seed(12200)
    points = torch.rand((192, 3), device="cuda", dtype=torch.float32)
    result = torchbvh.fps(points, 48, seed=3)
    distances = torch.cdist(points, result.points).square()
    expected_distance, expected_anchor = distances.min(-1)
    torch.testing.assert_close(result.nearest_anchor_dist_sq, expected_distance)
    torch.testing.assert_close(result.nearest_anchor.long(), expected_anchor)
    torch.testing.assert_close(
        result.anchor_counts,
        torch.bincount(expected_anchor, minlength=48).to(torch.int32),
    )
    for anchor in range(48):
        assigned = expected_anchor == anchor
        expected_radius = (
            expected_distance[assigned].max()
            if bool(assigned.any())
            else expected_distance.new_zeros(())
        )
        torch.testing.assert_close(result.anchor_radius[anchor], expected_radius)


def test_exact_full_scan_mode_has_been_removed():
    points = torch.rand((64, 3))
    with pytest.raises(ValueError, match="mode must be one of"):
        torchbvh.fps(points, 16, mode="exact_full_scan")


def test_exact_graph_and_nongraph_match():
    assert torch.cuda.is_available()
    torch.manual_seed(12300)
    points = torch.rand((2, 512, 3), device="cuda", dtype=torch.float32)
    graph = torchbvh.fps(points, 128, use_graph=True)
    eager = torchbvh.fps(points, 128, use_graph=False)
    for field in graph.__dataclass_fields__:
        torch.testing.assert_close(getattr(graph, field), getattr(eager, field))


def test_exact_graph_workspaces_are_stream_scoped():
    assert torch.cuda.is_available()
    torch.manual_seed(12400)
    first = torch.rand((2, 256, 3), device="cuda")
    second = torch.rand((2, 256, 3), device="cuda")
    expected_first = torchbvh.fps(first, 64, seed=1, use_graph=False)
    expected_second = torchbvh.fps(second, 64, seed=7, use_graph=False)

    stream_a, stream_b = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(stream_a):
        actual_first = torchbvh.fps(first, 64, seed=1, use_graph=True)
    with torch.cuda.stream(stream_b):
        actual_second = torchbvh.fps(second, 64, seed=7, use_graph=True)
    torch.cuda.synchronize()

    for field in expected_first.__dataclass_fields__:
        torch.testing.assert_close(
            getattr(actual_first, field), getattr(expected_first, field)
        )
        torch.testing.assert_close(
            getattr(actual_second, field), getattr(expected_second, field)
        )


def test_approximate_fps_returns_consistent_assignment_and_quality():
    assert torch.cuda.is_available()
    torch.manual_seed(12500)
    points = torch.rand((1024, 3), device="cuda", dtype=torch.float32)
    exact = torchbvh.fps(points, 256)
    approximate = torchbvh.fps(
        points,
        256,
        mode="approx_bucketed",
        r=4,
        c=2,
        alpha=0.25,
    )
    _assert_result(approximate, points, 256)
    distances = torch.cdist(points, approximate.points).square()
    expected_distance, expected_anchor = distances.min(-1)
    torch.testing.assert_close(
        approximate.nearest_anchor_dist_sq, expected_distance, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(approximate.nearest_anchor.long(), expected_anchor)
    mean_ratio = (
        approximate.nearest_anchor_dist_sq.sqrt().mean()
        / exact.nearest_anchor_dist_sq.sqrt().mean()
    )
    assert mean_ratio < 1.25


def test_fps_validation():
    assert torch.cuda.is_available()
    points = torch.rand((64, 3), device="cuda")
    with pytest.raises(ValueError, match="mode must be one of"):
        torchbvh.fps(points, 16, mode="bogus")
    with pytest.raises(ValueError, match="r must be"):
        torchbvh.fps(points, 16, mode="approx_bucketed", r=0)
    with pytest.raises(ValueError, match="c must be"):
        torchbvh.fps(points, 16, mode="approx_bucketed", c=0)
    with pytest.raises(ValueError, match=r"r \* c"):
        torchbvh.fps(points, 16, mode="approx_bucketed", r=8, c=5)
    with pytest.raises(ValueError, match="target token count"):
        torchbvh.fps(points, 0)
