import pytest
import torch

import torchbvh


def _offsets(values):
    return torch.tensor(values, device="cuda", dtype=torch.int64).contiguous()


def test_ragged_build_query_destroy_lifecycle_and_shapes():
    assert torch.cuda.is_available()
    torch.manual_seed(14001)
    counts = [9, 17, 12]
    query_counts = [4, 7, 5]
    points = torch.cat(
        [
            torch.rand((count, 3), device="cuda", dtype=torch.float32) + batch * 10.0
            for batch, count in enumerate(counts)
        ],
        dim=0,
    ).contiguous()
    query_points = torch.cat(
        [
            torch.rand((count, 3), device="cuda", dtype=torch.float32) + batch * 10.0
            for batch, count in enumerate(query_counts)
        ],
        dim=0,
    ).contiguous()
    batch_offsets = _offsets([0, 9, 26, 38])
    query_offsets = _offsets([0, 4, 11, 16])

    bvh = torchbvh.build_bvh_ragged(points, batch_offsets)

    assert isinstance(bvh, torchbvh.RaggedBVHHandle)
    assert not bvh.destroyed
    assert bvh["batch_size"] == 3
    assert bvh["dim"] == 3
    torch.testing.assert_close(bvh["batch_offsets"], batch_offsets)
    torch.testing.assert_close(
        bvh["num_leaves_per_sample"],
        torch.tensor(counts, device="cuda", dtype=torch.int64),
    )

    indices, distances, neighbor_positions = torchbvh.query_knn_ragged(
        bvh,
        query_points,
        query_offsets,
        4,
        source_points=points,
    )

    assert indices.shape == (16, 4)
    assert distances.shape == (16, 4)
    assert neighbor_positions.shape == (16, 4, 3)
    assert indices.dtype == torch.int64
    assert distances.dtype == torch.float32
    assert torch.all(distances[:, :-1] <= distances[:, 1:])
    for batch, count in enumerate(counts):
        q0, q1 = int(query_offsets[batch]), int(query_offsets[batch + 1])
        assert int(indices[q0:q1].min()) >= 0
        assert int(indices[q0:q1].max()) < count

    torchbvh.destroy_bvh(bvh)
    assert bvh.destroyed
    with pytest.raises(RuntimeError, match="destroyed"):
        torchbvh.query_knn_ragged(bvh, query_points, query_offsets, 4)
    with pytest.raises(RuntimeError, match="destroyed"):
        _ = bvh["dim"]


def test_ragged_query_rejects_other_handle_types():
    assert torch.cuda.is_available()
    points = torch.rand((16, 2), device="cuda", dtype=torch.float32).contiguous()
    query_points = points[:4].contiguous()
    offsets = _offsets([0, 16])
    single = torchbvh.build_bvh(points)
    batched = torchbvh.build_bvh_batched(points.view(1, 16, 2).contiguous())

    with pytest.raises(TypeError, match="RaggedBVHHandle"):
        torchbvh.query_knn_ragged(single, query_points, offsets, 4)
    with pytest.raises(TypeError, match="RaggedBVHHandle"):
        torchbvh.query_knn_ragged(batched, query_points, offsets, 4)


@pytest.mark.parametrize(
    "bad_offsets,match",
    [
        (lambda o: o.view(1, -1), "shape"),
        (lambda o: o.to(torch.int32), "int64"),
        (lambda o: o.cpu(), "CUDA"),
        (lambda o: torch.empty((4, 2), device="cuda", dtype=torch.int64)[:, 0], "contiguous"),
        (lambda o: _offsets([1, 4, 8]), "start at 0"),
        (lambda o: _offsets([0, 4, 7]), "final offset"),
        (lambda o: _offsets([0, 8, 8]), "strictly increasing"),
    ],
)
def test_ragged_build_rejects_bad_offsets(bad_offsets, match):
    assert torch.cuda.is_available()
    points = torch.rand((8, 2), device="cuda", dtype=torch.float32).contiguous()

    with pytest.raises(ValueError, match=match):
        torchbvh.build_bvh_ragged(points, bad_offsets(_offsets([0, 4, 8])))


@pytest.mark.parametrize(
    "bad_points,match",
    [
        (lambda p: p.view(1, 8, 2), "shape"),
        (lambda p: torch.rand((8, 4), device="cuda"), "D must be 2 or 3"),
        (lambda p: p.double(), "float32"),
        (lambda p: torch.empty((8, 3), device="cuda", dtype=p.dtype)[:, :2], "contiguous"),
        (lambda p: p.cpu(), "CUDA"),
    ],
)
def test_ragged_build_rejects_bad_points(bad_points, match):
    assert torch.cuda.is_available()
    points = torch.rand((8, 2), device="cuda", dtype=torch.float32).contiguous()
    offsets = _offsets([0, 4, 8])

    with pytest.raises(ValueError, match=match):
        torchbvh.build_bvh_ragged(bad_points(points), offsets)


@pytest.mark.parametrize(
    "bad_query,match",
    [
        (lambda q: q.view(1, 8, 2), "shape"),
        (lambda q: torch.rand((8, 3), device="cuda"), "second dimension"),
        (lambda q: q.double(), "float32"),
        (lambda q: torch.empty((8, 3), device="cuda", dtype=q.dtype)[:, :2], "contiguous"),
        (lambda q: q.cpu(), "CUDA"),
    ],
)
def test_ragged_query_rejects_bad_query_points(bad_query, match):
    assert torch.cuda.is_available()
    points = torch.rand((16, 2), device="cuda", dtype=torch.float32).contiguous()
    query_points = torch.rand((8, 2), device="cuda", dtype=torch.float32).contiguous()
    bvh = torchbvh.build_bvh_ragged(points, _offsets([0, 9, 16]))

    with pytest.raises(ValueError, match=match):
        torchbvh.query_knn_ragged(bvh, bad_query(query_points), _offsets([0, 4, 8]), 4)


def test_ragged_query_rejects_unsupported_k_batch_mismatch_and_too_few_points():
    assert torch.cuda.is_available()
    points = torch.rand((11, 2), device="cuda", dtype=torch.float32).contiguous()
    query_points = torch.rand((8, 2), device="cuda", dtype=torch.float32).contiguous()
    bvh = torchbvh.build_bvh_ragged(points, _offsets([0, 3, 11]))

    with pytest.raises(ValueError, match="k must be 4, 8, or 16"):
        torchbvh.query_knn_ragged(bvh, query_points, _offsets([0, 4, 8]), 5)
    with pytest.raises(ValueError, match="batch size"):
        torchbvh.query_knn_ragged(bvh, query_points, _offsets([0, 2, 5, 8]), 4)
    with pytest.raises(ValueError, match="at least k"):
        torchbvh.query_knn_ragged(bvh, query_points, _offsets([0, 4, 8]), 4)


def test_ragged_handle_cascade_destroy_marks_inner_handles():
    assert torch.cuda.is_available()
    points = torch.rand((20, 3), device="cuda", dtype=torch.float32).contiguous()
    batch_offsets = _offsets([0, 9, 20])
    bvh = torchbvh.build_bvh_ragged(points, batch_offsets)
    inner_handles = list(bvh._handles)
    assert len(inner_handles) == 2
    assert all(not h.destroyed for h in inner_handles)

    torchbvh.destroy_bvh(bvh)

    assert bvh.destroyed
    assert all(h.destroyed for h in inner_handles)
    with pytest.raises(RuntimeError, match="destroyed"):
        _ = bvh["dim"]


def test_ragged_handle_idempotent_double_destroy():
    assert torch.cuda.is_available()
    points = torch.rand((20, 3), device="cuda", dtype=torch.float32).contiguous()
    batch_offsets = _offsets([0, 9, 20])
    bvh = torchbvh.build_bvh_ragged(points, batch_offsets)
    torchbvh.destroy_bvh(bvh)
    assert bvh.destroyed
    torchbvh.destroy_bvh(bvh)
    assert bvh.destroyed


def test_ragged_query_rejects_bad_source_points():
    assert torch.cuda.is_available()
    points = torch.rand((16, 2), device="cuda", dtype=torch.float32).contiguous()
    query_points = torch.rand((8, 2), device="cuda", dtype=torch.float32).contiguous()
    bvh = torchbvh.build_bvh_ragged(points, _offsets([0, 8, 16]))

    with pytest.raises(ValueError, match="first dimension"):
        torchbvh.query_knn_ragged(
            bvh,
            query_points,
            _offsets([0, 4, 8]),
            4,
            source_points=points[:-1].contiguous(),
        )
    with pytest.raises(ValueError, match="float32"):
        torchbvh.query_knn_ragged(
            bvh,
            query_points,
            _offsets([0, 4, 8]),
            4,
            source_points=points.double(),
        )
