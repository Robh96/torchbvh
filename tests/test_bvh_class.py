import pytest
import torch

import torchbvh
from torchbvh import BVH


def _make_points(n=20, dim=3):
    base = torch.linspace(-1.0, 1.0, n, device="cuda", dtype=torch.float32)
    if dim == 2:
        return torch.stack((base, base.square() - 0.25), dim=1).contiguous()
    return torch.stack((base, base.square() - 0.25, torch.sin(2.0 * base)), dim=1).contiguous()


def _make_features(n=20, f=4):
    return torch.randn((n, f), device="cuda", dtype=torch.float32)


# ---------------------------------------------------------------------------
# Export and __all__ membership
# ---------------------------------------------------------------------------


def test_bvh_classes_exported_from_package():
    assert hasattr(torchbvh, "BVH")
    assert not hasattr(torchbvh, "BatchedBVH")
    assert not hasattr(torchbvh, "RaggedBVH")


def test_bvh_classes_in_all():
    assert "BVH" in torchbvh.__all__
    assert "BatchedBVH" not in torchbvh.__all__
    assert "RaggedBVH" not in torchbvh.__all__


# ---------------------------------------------------------------------------
# BVH Ã¢â‚¬â€ no Mapping / __getitem__
# ---------------------------------------------------------------------------


def test_bvh_does_not_expose_getitem():
    assert torch.cuda.is_available()
    bvh = BVH(_make_points())
    assert not hasattr(bvh, "__getitem__")
    bvh.destroy()


def test_batched_bvh_does_not_expose_getitem():
    assert torch.cuda.is_available()
    points = _make_points().unsqueeze(0).expand(2, -1, -1).contiguous()
    bvh = BVH(points)
    assert not hasattr(bvh, "__getitem__")
    bvh.destroy()


def test_ragged_bvh_does_not_expose_getitem():
    assert torch.cuda.is_available()
    points = _make_points()
    offsets = torch.tensor([0, 10, 20], device="cuda", dtype=torch.int64)
    bvh = BVH(points, batch_offsets=offsets)
    assert not hasattr(bvh, "__getitem__")
    bvh.destroy()


# ---------------------------------------------------------------------------
# BVH Ã¢â‚¬â€ lifecycle
# ---------------------------------------------------------------------------


def test_bvh_destroyed_is_false_before_destroy():
    assert torch.cuda.is_available()
    bvh = BVH(_make_points())
    assert not bvh.destroyed
    bvh.destroy()


def test_bvh_destroyed_is_true_after_destroy():
    assert torch.cuda.is_available()
    bvh = BVH(_make_points())
    bvh.destroy()
    assert bvh.destroyed


def test_bvh_destroy_is_idempotent():
    assert torch.cuda.is_available()
    bvh = BVH(_make_points())
    bvh.destroy()
    bvh.destroy()  # must not raise


def test_bvh_context_manager_destroys_on_exit():
    assert torch.cuda.is_available()
    with BVH(_make_points()) as bvh:
        assert not bvh.destroyed
    assert bvh.destroyed


def test_bvh_knn_after_destroy_raises():
    assert torch.cuda.is_available()
    points = _make_points()
    bvh = BVH(points)
    bvh.destroy()
    with pytest.raises(RuntimeError):
        bvh.knn(points[:3], k=4)


# ---------------------------------------------------------------------------
# BVH Ã¢â‚¬â€ knn delegation
# ---------------------------------------------------------------------------


def test_bvh_knn_matches_query_knn():
    assert torch.cuda.is_available()
    points = _make_points(20, 3)
    queries = points[3:8] + 0.02
    k = 4

    with BVH(points) as bvh:
        idx_cls, dist_cls = bvh.knn(queries, k)

    bvh_handle = torchbvh.build_bvh(points)
    idx_fn, dist_fn = torchbvh.query_knn(bvh_handle, queries, k)
    torchbvh.destroy_bvh(bvh_handle)

    torch.testing.assert_close(idx_cls, idx_fn)
    torch.testing.assert_close(dist_cls, dist_fn)


def test_bvh_knn_with_source_points_matches_query_knn():
    assert torch.cuda.is_available()
    points = _make_points(20, 3)
    queries = points[3:8] + 0.02
    k = 4

    with BVH(points) as bvh:
        idx_cls, dist_cls, pos_cls = bvh.knn(queries, k, source_points=points)

    bvh_handle = torchbvh.build_bvh(points)
    idx_fn, dist_fn, pos_fn = torchbvh.query_knn(bvh_handle, queries, k, source_points=points)
    torchbvh.destroy_bvh(bvh_handle)

    torch.testing.assert_close(idx_cls, idx_fn)
    torch.testing.assert_close(dist_cls, dist_fn)
    torch.testing.assert_close(pos_cls, pos_fn)


# ---------------------------------------------------------------------------
# BVH Ã¢â‚¬â€ interpolate delegation
# ---------------------------------------------------------------------------


def test_bvh_interpolate_matches_bvh_mls_interpolate():
    assert torch.cuda.is_available()
    points = _make_points(20, 3)
    displaced = (points[4:9] + 0.03).contiguous()
    features = _make_features(20, 4)
    k = 8

    with BVH(points) as bvh:
        result_cls = bvh.interpolate(displaced, features, k)

    result_fn = torchbvh.mls_interpolate(points, displaced, features, k)

    assert isinstance(result_cls, torch.Tensor)
    torch.testing.assert_close(result_cls, result_fn, rtol=1e-5, atol=1e-5)


def test_bvh_interpolate_return_grad_true_returns_two_tuple():
    assert torch.cuda.is_available()
    points = _make_points(20, 3)
    displaced = (points[4:9] + 0.03).contiguous()
    features = _make_features(20, 4)

    with BVH(points) as bvh:
        result = bvh.interpolate(displaced, features, k=8, return_grad=True)

    assert isinstance(result, tuple)
    assert len(result) == 2
    interpolated, field_gradient = result
    assert interpolated.shape == (5, 4)
    assert field_gradient.shape == (5, 3, 4)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# BatchedBVH Ã¢â‚¬â€ lifecycle
# ---------------------------------------------------------------------------


def test_batched_bvh_destroyed_is_false_before_destroy():
    assert torch.cuda.is_available()
    points = _make_points().unsqueeze(0).expand(2, -1, -1).contiguous()
    bvh = BVH(points)
    assert not bvh.destroyed
    bvh.destroy()


def test_batched_bvh_context_manager_destroys_on_exit():
    assert torch.cuda.is_available()
    points = _make_points().unsqueeze(0).expand(2, -1, -1).contiguous()
    with BVH(points) as bvh:
        assert not bvh.destroyed
    assert bvh.destroyed


def test_batched_bvh_destroy_is_idempotent():
    assert torch.cuda.is_available()
    points = _make_points().unsqueeze(0).expand(2, -1, -1).contiguous()
    bvh = BVH(points)
    bvh.destroy()
    bvh.destroy()


# ---------------------------------------------------------------------------
# BatchedBVH Ã¢â‚¬â€ knn delegation
# ---------------------------------------------------------------------------


def test_batched_bvh_knn_matches_query_knn_batched():
    assert torch.cuda.is_available()
    points_single = _make_points(20, 3)
    points = points_single.unsqueeze(0).expand(2, -1, -1).contiguous()
    queries = (points[:, 3:8, :] + 0.02).contiguous()
    k = 4

    with BVH(points) as bvh:
        idx_cls, dist_cls = bvh.knn(queries, k)

    bvh_handle = torchbvh.build_bvh(points)
    idx_fn, dist_fn = torchbvh.query_knn(bvh_handle, queries, k)
    torchbvh.destroy_bvh(bvh_handle)

    torch.testing.assert_close(idx_cls, idx_fn)
    torch.testing.assert_close(dist_cls, dist_fn)


# ---------------------------------------------------------------------------
# RaggedBVH Ã¢â‚¬â€ lifecycle
# ---------------------------------------------------------------------------


def test_ragged_bvh_destroyed_is_false_before_destroy():
    assert torch.cuda.is_available()
    points = _make_points(20)
    offsets = torch.tensor([0, 10, 20], device="cuda", dtype=torch.int64)
    bvh = BVH(points, batch_offsets=offsets)
    assert not bvh.destroyed
    bvh.destroy()


def test_ragged_bvh_context_manager_destroys_on_exit():
    assert torch.cuda.is_available()
    points = _make_points(20)
    offsets = torch.tensor([0, 10, 20], device="cuda", dtype=torch.int64)
    with BVH(points, batch_offsets=offsets) as bvh:
        assert not bvh.destroyed
    assert bvh.destroyed


# ---------------------------------------------------------------------------
# RaggedBVH Ã¢â‚¬â€ knn delegation
# ---------------------------------------------------------------------------


def test_ragged_bvh_knn_matches_query_knn_ragged():
    assert torch.cuda.is_available()
    points = _make_points(20, 3)
    src_offsets = torch.tensor([0, 10, 20], device="cuda", dtype=torch.int64)
    queries = (points[:8] + 0.02).contiguous()
    q_offsets = torch.tensor([0, 4, 8], device="cuda", dtype=torch.int64)
    k = 4

    with BVH(points, batch_offsets=src_offsets) as bvh:
        idx_cls, dist_cls = bvh.knn(queries, k, query_offsets=q_offsets)

    bvh_handle = torchbvh.build_bvh(points, batch_offsets=src_offsets)
    idx_fn, dist_fn = torchbvh.query_knn(bvh_handle, queries, k, query_offsets=q_offsets)
    torchbvh.destroy_bvh(bvh_handle)

    torch.testing.assert_close(idx_cls, idx_fn)
    torch.testing.assert_close(dist_cls, dist_fn)


# ---------------------------------------------------------------------------
# RaggedBVH Ã¢â‚¬â€ unsupported methods raise TypeError
# ---------------------------------------------------------------------------


def test_ragged_bvh_interpolate_raises_type_error():
    assert torch.cuda.is_available()
    points = _make_points(20)
    offsets = torch.tensor([0, 10, 20], device="cuda", dtype=torch.int64)
    with BVH(points, batch_offsets=offsets) as bvh:
        with pytest.raises(TypeError, match="interpolate"):
            bvh.interpolate(points[:5], _make_features(20))
