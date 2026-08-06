import pytest
import torch

import torchbvh
import torchbvh._query as query_module
import torchbvh._reorder as reorder_module


@pytest.mark.parametrize("dim", [2, 3])
def test_python_api_build_query_destroy_lifecycle(dim):
    assert torch.cuda.is_available()
    torch.manual_seed(7000 + dim)
    points = torch.rand((24, dim), device="cuda", dtype=torch.float32)
    query_points = points[:6] + 0.01

    bvh = torchbvh.build_bvh(points.contiguous())

    assert isinstance(bvh, torchbvh.BVHHandle)
    assert not bvh.destroyed
    assert bvh["dim"] == dim

    indices, distances = torchbvh.query_knn(bvh, query_points.contiguous(), 4)
    expected_distances = torch.cdist(query_points, points).square().topk(4, largest=False).values

    assert indices.shape == (query_points.shape[0], 4)
    assert distances.shape == (query_points.shape[0], 4)
    assert torch.all(indices >= 0)
    assert torch.all(indices < points.shape[0])
    assert torch.all(distances[:, :-1] <= distances[:, 1:])
    torch.testing.assert_close(distances, expected_distances, rtol=1.0e-5, atol=1.0e-5)

    torchbvh.destroy_bvh(bvh)
    assert bvh.destroyed
    with pytest.raises(RuntimeError, match="destroyed"):
        torchbvh.query_knn(bvh, query_points.contiguous(), 4)
    with pytest.raises(RuntimeError, match="destroyed"):
        _ = bvh["dim"]


def test_python_api_rejects_unsupported_k_with_public_message():
    assert torch.cuda.is_available()
    points = torch.rand((8, 2), device="cuda", dtype=torch.float32)
    bvh = torchbvh.build_bvh(points.contiguous())

    with pytest.raises(ValueError, match="k must be 4, 8, or 16"):
        torchbvh.query_knn(bvh, points.contiguous(), 5)


def test_python_api_preserves_legacy_mapping_query_path():
    assert torch.cuda.is_available()
    points = torch.rand((16, 3), device="cuda", dtype=torch.float32)
    bvh = torchbvh.build_bvh(points.contiguous())
    legacy_bvh = dict(bvh)

    indices, distances = torchbvh.query_knn(legacy_bvh, points[:3].contiguous(), 4)

    assert indices.shape == (3, 4)
    assert distances.shape == (3, 4)


def test_query_knn_sort_queries_true_uses_ordered_traversal(monkeypatch):
    assert torch.cuda.is_available()
    points = torch.rand((16, 3), device="cuda", dtype=torch.float32).contiguous()
    queries = points[:5].contiguous()
    bvh = torchbvh.build_bvh(points)
    calls = []

    def fake_sort_queries(query_batch, scene_min, scene_max):
        calls.append(("sort", query_batch.shape, scene_min.shape, scene_max.shape))
        batch_size, query_count, _ = query_batch.shape
        perm = torch.arange(query_count, device=query_batch.device, dtype=torch.int64).expand(batch_size, query_count)
        return perm.contiguous(), perm.contiguous()

    def fake_ordered(*args):
        calls.append(("ordered", args[2].shape, args[3].shape, args[-1]))
        query_count = args[2].size(0)
        k = int(args[-1])
        return (
            torch.zeros((query_count, k), device=args[2].device, dtype=torch.int64),
            torch.zeros((query_count, k), device=args[2].device, dtype=torch.float32),
        )

    def fake_unordered(*args):
        raise AssertionError("unordered query route should not be used when sort_queries=True")

    monkeypatch.setattr(reorder_module, "morton_sort_queries_batched", fake_sort_queries)
    monkeypatch.setattr(query_module._C, "query_knn_ordered", fake_ordered)
    monkeypatch.setattr(query_module._C, "query_knn", fake_unordered)

    indices, distances = torchbvh.query_knn(bvh, queries, 4, sort_queries=True)

    assert indices.shape == (5, 4)
    assert distances.shape == (5, 4)
    assert [call[0] for call in calls] == ["sort", "ordered"]
    assert calls[0][1] == (1, 5, 3)
    assert calls[1][1] == (5, 3)
    assert calls[1][2] == (5,)
    assert calls[1][3] == 4


def test_python_api_rejects_invalid_handle_type():
    with pytest.raises(TypeError, match="BVHHandle or mapping"):
        torchbvh.query_knn(object(), torch.empty((0, 2)), 4)


def test_bvh_handle_idempotent_double_destroy():
    assert torch.cuda.is_available()
    points = torch.rand((12, 3), device="cuda", dtype=torch.float32).contiguous()
    bvh = torchbvh.build_bvh(points)
    torchbvh.destroy_bvh(bvh)
    assert bvh.destroyed
    torchbvh.destroy_bvh(bvh)
    assert bvh.destroyed


def test_destroy_bvh_accepts_legacy_plain_dict():
    assert torch.cuda.is_available()
    points = torch.rand((12, 3), device="cuda", dtype=torch.float32).contiguous()
    bvh = torchbvh.build_bvh(points)
    legacy = dict(bvh)
    assert not legacy.get("_destroyed", False)
    torchbvh.destroy_bvh(legacy)
    assert legacy.get("_destroyed") is True
    with pytest.raises(RuntimeError, match="destroyed"):
        torchbvh.query_knn(legacy, points[:3].contiguous(), 4)


def test_bvh_handles_have_traversal_fields():
    assert torch.cuda.is_available()
    points = torch.rand((16, 3), device="cuda", dtype=torch.float32).contiguous()
    single = torchbvh.build_bvh(points)
    for key in ("left_child_mem", "mem_to_leaf"):
        assert key in single, f"BVHHandle missing key: {key}"
        assert single[key].dtype == torch.int32
    batched_pts = torch.rand((2, 16, 3), device="cuda", dtype=torch.float32).contiguous()
    batched = torchbvh.build_bvh_batched(batched_pts)
    for key in ("left_child_mem", "mem_to_leaf"):
        assert key in batched, f"BatchedBVHHandle missing key: {key}"
        assert batched[key].dtype == torch.int32


def test_supported_constants_importable_from_package():
    assert torchbvh.SUPPORTED_K == (4, 8, 16)
    assert torchbvh.SUPPORTED_DIMS == (2, 3)
    assert "SUPPORTED_K" in torchbvh.__all__
    assert "SUPPORTED_DIMS" in torchbvh.__all__


def test_public_package_surface_after_cleanup():
    import torchbvh as ibvh

    expected_public = {
        "BVH",
        "fps",
        "FPSResult",
        "build_bvh",
        "query_knn",
        "mls_interpolate",
        "bvh_mls_interpolate_batched_heads",
        "SUPPORTED_K",
        "SUPPORTED_DIMS",
    }
    expected_compatibility = {
        "BatchedBVH",
        "RaggedBVH",
        "build_bvh_batched",
        "query_knn_batched",
        "bvh_mls_interpolate_batched",
    }
    removed = {
        "_linear_mls",
        "_linear_mls_batched",
    }

    exported = set(ibvh.__all__)
    assert expected_public <= exported
    assert expected_compatibility <= exported
    assert removed.isdisjoint(exported)
    for name in removed:
        assert not hasattr(ibvh, name)


def test_recommended_public_workflow_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for torchbvh workflow smoke test")

    import torchbvh as ibvh

    grid = torch.linspace(0.0, 1.0, 4, device="cuda", dtype=torch.float32)
    points = torch.stack((grid.repeat_interleave(4), grid.repeat(4)), dim=1).contiguous()
    query_points = (points[:6] + torch.tensor([0.01, -0.02], device="cuda")).contiguous()
    displaced_points = (points[4:10] + torch.tensor([0.02, 0.015], device="cuda")).contiguous()
    features = torch.stack(
        (points[:, 0], points[:, 1], points[:, 0] + points[:, 1]),
        dim=1,
    ).contiguous()

    with ibvh.BVH(points) as bvh:
        indices, squared_distances = bvh.knn(query_points, k=4)
        interpolated = bvh.interpolate(displaced_points, features, k=4)
        assert not bvh.destroyed

    assert bvh.destroyed
    assert indices.shape == (query_points.shape[0], 4)
    assert squared_distances.shape == (query_points.shape[0], 4)
    assert interpolated.shape == (displaced_points.shape[0], features.shape[1])
    assert torch.all(indices >= 0)
    assert torch.all(indices < points.shape[0])
    assert torch.all(squared_distances[:, :-1] <= squared_distances[:, 1:])
    assert torch.isfinite(interpolated).all()

    fps_result = ibvh.fps(points, target_tokens=4)
    assert isinstance(fps_result, ibvh.FPSResult)
    assert fps_result.indices.shape == (4,)
    assert fps_result.points.shape == (4, 2)
    assert fps_result.nearest_anchor.shape == (points.shape[0],)
    assert torch.equal(fps_result.selection_order_indices, fps_result.indices)
