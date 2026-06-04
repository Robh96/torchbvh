import pytest
import torch

import torchbvh


def test_batched_build_query_destroy_lifecycle_and_shapes():
    assert torch.cuda.is_available()
    torch.manual_seed(13001)
    points = torch.rand((3, 17, 3), device="cuda", dtype=torch.float32).contiguous()
    query_points = torch.rand((3, 5, 3), device="cuda", dtype=torch.float32).contiguous()

    bvh = torchbvh.build_bvh_batched(points)

    assert isinstance(bvh, torchbvh.BatchedBVHHandle)
    assert not bvh.destroyed
    assert bvh["batch_size"] == 3
    assert bvh["num_leaves"] == 17
    assert bvh["dim"] == 3
    assert bvh["node_aabbs"].shape == (3, bvh["num_real_nodes"], 6)
    assert bvh["sorted_indices"].shape == (3, 17)
    assert bvh["scene_min"].shape == (3, 3)
    assert bvh["scene_max"].shape == (3, 3)

    expected = torch.arange(17, device="cuda")
    for batch in range(3):
        torch.testing.assert_close(bvh["sorted_indices"][batch].sort().values, expected)

    indices, distances, neighbor_positions = torchbvh.query_knn_batched(
        bvh,
        query_points,
        4,
        source_points=points,
    )
    assert indices.shape == (3, 5, 4)
    assert distances.shape == (3, 5, 4)
    assert neighbor_positions.shape == (3, 5, 4, 3)
    assert torch.all(indices >= 0)
    assert torch.all(indices < 17)
    assert torch.all(distances[:, :, :-1] <= distances[:, :, 1:])
    torch.testing.assert_close(
        neighbor_positions,
        torch.gather(
            points.unsqueeze(1).expand(-1, 5, -1, -1),
            2,
            indices.unsqueeze(-1).expand(-1, -1, -1, 3),
        ),
    )

    torchbvh.destroy_bvh(bvh)
    assert bvh.destroyed
    with pytest.raises(RuntimeError, match="destroyed"):
        torchbvh.query_knn_batched(bvh, query_points, 4)
    with pytest.raises(RuntimeError, match="destroyed"):
        _ = bvh["dim"]


def test_batched_api_accepts_legacy_mapping_and_rejects_single_sample_handle():
    assert torch.cuda.is_available()
    points = torch.rand((2, 16, 2), device="cuda", dtype=torch.float32).contiguous()
    query_points = points[:, :3, :].contiguous()
    bvh = torchbvh.build_bvh_batched(points)
    legacy_bvh = dict(bvh)

    indices, distances = torchbvh.query_knn_batched(legacy_bvh, query_points, 4)
    assert indices.shape == (2, 3, 4)
    assert distances.shape == (2, 3, 4)

    single = torchbvh.build_bvh(points[0].contiguous())
    with pytest.raises(TypeError, match="BatchedBVHHandle"):
        torchbvh.query_knn_batched(single, query_points, 4)
    # Unified query_knn routes to the batched path; wrong-rank query raises ValueError.
    with pytest.raises(ValueError, match="shape"):
        torchbvh.query_knn(bvh, query_points[0].contiguous(), 4)


@pytest.mark.parametrize(
    "bad_query,match",
    [
        (lambda q: q[0], "shape"),
        (lambda q: q[:1], "batch size"),
        (lambda q: torch.rand((2, 4, 3), device="cuda"), "last dimension"),
        (lambda q: q.double(), "float32"),
        (lambda q: torch.empty((2, 4, 4), device="cuda", dtype=q.dtype)[:, :, :2], "contiguous"),
    ],
)
def test_batched_query_validation_errors(bad_query, match):
    assert torch.cuda.is_available()
    points = torch.rand((2, 16, 2), device="cuda", dtype=torch.float32).contiguous()
    query_points = torch.rand((2, 4, 2), device="cuda", dtype=torch.float32).contiguous()
    bvh = torchbvh.build_bvh_batched(points)

    with pytest.raises((ValueError, RuntimeError), match=match):
        torchbvh.query_knn_batched(bvh, bad_query(query_points), 4)


def test_batched_query_rejects_unsupported_k_with_public_message():
    assert torch.cuda.is_available()
    points = torch.rand((2, 16, 2), device="cuda", dtype=torch.float32).contiguous()
    bvh = torchbvh.build_bvh_batched(points)

    with pytest.raises(ValueError, match="k must be 4, 8, or 16"):
        torchbvh.query_knn_batched(bvh, points, 5)


def test_batched_handle_idempotent_double_destroy():
    assert torch.cuda.is_available()
    points = torch.rand((2, 12, 3), device="cuda", dtype=torch.float32).contiguous()
    bvh = torchbvh.build_bvh_batched(points)
    torchbvh.destroy_bvh(bvh)
    assert bvh.destroyed
    torchbvh.destroy_bvh(bvh)
    assert bvh.destroyed


def test_batched_build_rejects_wrong_rank_dtype_device_and_noncontiguous():
    assert torch.cuda.is_available()
    points = torch.rand((2, 16, 2), device="cuda", dtype=torch.float32).contiguous()

    with pytest.raises(RuntimeError, match="shape"):
        torchbvh.build_bvh_batched(points[0])
    with pytest.raises(RuntimeError, match="float32"):
        torchbvh.build_bvh_batched(points.double())
    with pytest.raises(RuntimeError, match="contiguous"):
        torchbvh.build_bvh_batched(points.transpose(0, 1))
    with pytest.raises(RuntimeError, match="CUDA"):
        torchbvh.build_bvh_batched(points.cpu())
