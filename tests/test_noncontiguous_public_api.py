import pytest
import torch

import torchbvh
from torchbvh import BVH, BatchedBVH, RaggedBVH


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _noncontiguous_float(shape, *, fill=None):
    storage_shape = (*shape[:-1], shape[-1] + 1)
    storage = torch.empty(storage_shape, device="cuda", dtype=torch.float32)
    view = storage[..., : shape[-1]]
    if fill is None:
        view.copy_(torch.randn(shape, device="cuda", dtype=torch.float32))
    else:
        view.copy_(fill)
    assert not view.is_contiguous()
    return view


def _noncontiguous_int64(values):
    values = torch.tensor(values, device="cuda", dtype=torch.int64)
    storage = torch.empty((values.numel(), 2), device="cuda", dtype=torch.int64)
    storage[:, 0] = values
    view = storage[:, 0]
    assert not view.is_contiguous()
    return view


def test_single_bvh_query_accepts_noncontiguous_points_queries_and_source_points():
    points = _noncontiguous_float((32, 3))
    queries = _noncontiguous_float((7, 3), fill=points[:7] + 0.01)

    bvh = torchbvh.build_bvh(points)
    actual = torchbvh.query_knn(bvh, queries, 4, source_points=points)

    expected_bvh = torchbvh.build_bvh(points.contiguous())
    expected = torchbvh.query_knn(
        expected_bvh,
        queries.contiguous(),
        4,
        source_points=points.contiguous(),
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)
    torchbvh.destroy_bvh(bvh)
    torchbvh.destroy_bvh(expected_bvh)


def test_batched_bvh_query_accepts_noncontiguous_points_queries_and_source_points():
    points = _noncontiguous_float((2, 32, 3))
    queries = _noncontiguous_float((2, 7, 3), fill=points[:, :7, :] + 0.01)

    bvh = torchbvh.build_bvh_batched(points)
    actual = torchbvh.query_knn_batched(bvh, queries, 4, source_points=points)

    expected_bvh = torchbvh.build_bvh_batched(points.contiguous())
    expected = torchbvh.query_knn_batched(
        expected_bvh,
        queries.contiguous(),
        4,
        source_points=points.contiguous(),
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)
    torchbvh.destroy_bvh(bvh)
    torchbvh.destroy_bvh(expected_bvh)


def test_ragged_bvh_query_accepts_noncontiguous_points_queries_offsets_and_source_points():
    points = _noncontiguous_float((24, 2))
    queries = _noncontiguous_float((10, 2), fill=points[:10] + 0.01)
    point_offsets = _noncontiguous_int64([0, 9, 24])
    query_offsets = _noncontiguous_int64([0, 4, 10])

    bvh = torchbvh.build_bvh_ragged(points, point_offsets)
    actual = torchbvh.query_knn_ragged(
        bvh,
        queries,
        query_offsets,
        4,
        source_points=points,
    )

    expected_bvh = torchbvh.build_bvh_ragged(points.contiguous(), point_offsets.contiguous())
    expected = torchbvh.query_knn_ragged(
        expected_bvh,
        queries.contiguous(),
        query_offsets.contiguous(),
        4,
        source_points=points.contiguous(),
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)
    torchbvh.destroy_bvh(bvh)
    torchbvh.destroy_bvh(expected_bvh)


def test_single_mls_accepts_noncontiguous_inputs_and_preserves_gradients():
    points = _noncontiguous_float((32, 3))
    displaced_storage = torch.empty((9, 4), device="cuda", dtype=torch.float32, requires_grad=True)
    displaced = displaced_storage[:, :3]
    displaced.data.copy_(points[:9] + 0.02)
    feature_storage = torch.randn((32, 6), device="cuda", dtype=torch.float32, requires_grad=True)
    features = feature_storage[:, :5]
    assert not displaced.is_contiguous()
    assert not features.is_contiguous()

    actual = torchbvh.bvh_mls_interpolate(points, displaced, features, k=8)
    expected = torchbvh.bvh_mls_interpolate(
        points.contiguous(),
        displaced.contiguous(),
        features.contiguous(),
        k=8,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    actual.square().mean().backward()
    assert displaced_storage.grad is not None
    assert feature_storage.grad is not None
    assert torch.isfinite(displaced_storage.grad).all()
    assert torch.isfinite(feature_storage.grad).all()


def test_batched_mls_accepts_noncontiguous_inputs_and_preserves_gradients():
    points = _noncontiguous_float((2, 32, 3))
    displaced_storage = torch.empty((2, 9, 4), device="cuda", dtype=torch.float32, requires_grad=True)
    displaced = displaced_storage[..., :3]
    displaced.data.copy_(points[:, :9, :] + 0.02)
    feature_storage = torch.randn((2, 32, 6), device="cuda", dtype=torch.float32, requires_grad=True)
    features = feature_storage[..., :5]
    assert not displaced.is_contiguous()
    assert not features.is_contiguous()

    actual = torchbvh.bvh_mls_interpolate_batched(points, displaced, features, k=8)
    expected = torchbvh.bvh_mls_interpolate_batched(
        points.contiguous(),
        displaced.contiguous(),
        features.contiguous(),
        k=8,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    actual.square().mean().backward()
    assert displaced_storage.grad is not None
    assert feature_storage.grad is not None
    assert torch.isfinite(displaced_storage.grad).all()
    assert torch.isfinite(feature_storage.grad).all()


def test_displaced_multihead_interpolation_accepts_noncontiguous_inputs_and_preserves_gradients():
    pos = _noncontiguous_float((2, 24, 3))
    q = _noncontiguous_float((2, 24, 3, 3), fill=pos[:, :, None, :] + 0.01)
    value_storage = torch.randn((2, 24, 3, 6), device="cuda", dtype=torch.float32, requires_grad=True)
    values = value_storage[..., :5]
    assert not values.is_contiguous()

    actual = torchbvh.interpolate_displaced(pos, q, values, k=4)
    expected = torchbvh.interpolate_displaced(
        pos.contiguous(),
        q.contiguous(),
        values.contiguous(),
        k=4,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    actual.square().mean().backward()
    assert value_storage.grad is not None
    assert torch.isfinite(value_storage.grad).all()


def test_gather_neighbor_values_accepts_noncontiguous_values_and_indices():
    values = _noncontiguous_float((2, 12, 3, 5))
    indices = _noncontiguous_int64([0, 1, 2, 3]).view(1, 1, 1, 4).expand(2, 12, 3, 4)
    assert not indices.is_contiguous()

    actual = torchbvh.gather_neighbor_values(values, indices)
    expected = torchbvh.gather_neighbor_values(values.contiguous(), indices.contiguous())
    torch.testing.assert_close(actual, expected)


def test_bvh_classes_store_contiguous_points_for_later_operations():
    points = _noncontiguous_float((32, 3))
    displaced = _noncontiguous_float((9, 3), fill=points[:9] + 0.02)
    features = _noncontiguous_float((32, 5))

    with BVH(points) as bvh:
        actual = bvh.interpolate(displaced, features, k=8)

    expected = torchbvh.bvh_mls_interpolate(
        points.contiguous(),
        displaced.contiguous(),
        features.contiguous(),
        k=8,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    batched_points = _noncontiguous_float((2, 32, 3))
    batched_queries = _noncontiguous_float((2, 7, 3), fill=batched_points[:, :7, :] + 0.01)
    with BatchedBVH(batched_points) as bvh:
        idx, dist = bvh.knn(batched_queries, 4)
    expected_bvh = torchbvh.build_bvh_batched(batched_points.contiguous())
    expected_idx, expected_dist = torchbvh.query_knn_batched(expected_bvh, batched_queries.contiguous(), 4)
    torch.testing.assert_close(idx, expected_idx)
    torch.testing.assert_close(dist, expected_dist)
    torchbvh.destroy_bvh(expected_bvh)

    ragged_points = _noncontiguous_float((24, 2))
    offsets = _noncontiguous_int64([0, 9, 24])
    q_offsets = _noncontiguous_int64([0, 4, 10])
    ragged_queries = _noncontiguous_float((10, 2), fill=ragged_points[:10] + 0.01)
    with RaggedBVH(ragged_points, offsets) as bvh:
        idx, dist = bvh.knn(ragged_queries, 4, query_offsets=q_offsets)
    expected_bvh = torchbvh.build_bvh_ragged(ragged_points.contiguous(), offsets.contiguous())
    expected_idx, expected_dist = torchbvh.query_knn_ragged(
        expected_bvh,
        ragged_queries.contiguous(),
        q_offsets.contiguous(),
        4,
    )
    torch.testing.assert_close(idx, expected_idx)
    torch.testing.assert_close(dist, expected_dist)
    torchbvh.destroy_bvh(expected_bvh)


def test_fps_accepts_noncontiguous_single_and_batched_points():
    points = _noncontiguous_float((32, 3))
    actual = torchbvh.fps(points, 8, mode="exact_full_scan")
    expected = torchbvh.fps(points.contiguous(), 8, mode="exact_full_scan")
    torch.testing.assert_close(actual.indices, expected.indices)
    torch.testing.assert_close(actual.points, expected.points)

    batched = _noncontiguous_float((2, 32, 3))
    actual_batched = torchbvh.fps(batched, 8, mode="exact_full_scan")
    expected_batched = torchbvh.fps(batched.contiguous(), 8, mode="exact_full_scan")
    torch.testing.assert_close(actual_batched.indices, expected_batched.indices)
    torch.testing.assert_close(actual_batched.points, expected_batched.points)
