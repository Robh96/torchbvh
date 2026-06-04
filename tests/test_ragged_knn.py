import pytest
import torch

import torchbvh


def _offsets(lengths):
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return torch.tensor(values, device="cuda", dtype=torch.int64).contiguous()


def _assert_ragged_matches_single_loop(
    points,
    point_counts,
    queries,
    query_counts,
    k,
    *,
    check_loop_indices=True,
):
    batch_offsets = _offsets(point_counts)
    query_offsets = _offsets(query_counts)
    bvh = torchbvh.build_bvh_ragged(points.contiguous(), batch_offsets)
    indices, distances, positions = torchbvh.query_knn_ragged(
        bvh,
        queries.contiguous(),
        query_offsets,
        k,
        source_points=points.contiguous(),
    )

    loop_indices = []
    loop_distances = []
    loop_positions = []
    loop_feature_gathers = []
    features = torch.arange(
        points.size(0) * 5,
        device=points.device,
        dtype=torch.float32,
    ).reshape(points.size(0), 5)
    for batch in range(len(point_counts)):
        p0, p1 = int(batch_offsets[batch]), int(batch_offsets[batch + 1])
        q0, q1 = int(query_offsets[batch]), int(query_offsets[batch + 1])
        single_points = points[p0:p1].contiguous()
        single_queries = queries[q0:q1].contiguous()
        single = torchbvh.build_bvh(single_points)
        expected_indices, expected_distances, expected_positions = torchbvh.query_knn(
            single,
            single_queries,
            k,
            source_points=single_points,
        )
        loop_indices.append(expected_indices)
        loop_distances.append(expected_distances)
        loop_positions.append(expected_positions)
        loop_feature_gathers.append(features[p0:p1][expected_indices])

    assert indices.shape == (queries.size(0), k)
    assert distances.shape == (queries.size(0), k)
    assert positions.shape == (queries.size(0), k, points.size(1))
    if check_loop_indices:
        torch.testing.assert_close(indices, torch.cat(loop_indices), rtol=0, atol=0)
    torch.testing.assert_close(distances, torch.cat(loop_distances), rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(positions, torch.cat(loop_positions), rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(
        distances,
        (positions - queries[:, None, :]).square().sum(dim=-1),
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    assert torch.all(distances[:, :-1] <= distances[:, 1:])

    start = 0
    gathered_features = []
    for batch, (point_count, query_count) in enumerate(zip(point_counts, query_counts)):
        p0, p1 = int(batch_offsets[batch]), int(batch_offsets[batch + 1])
        sample_indices = indices[start : start + query_count]
        assert int(sample_indices.min()) >= 0
        assert int(sample_indices.max()) < point_count
        gathered_features.append(features[p0:p1][sample_indices])
        start += query_count
    gathered_features = torch.cat(gathered_features)
    assert gathered_features.shape == (queries.size(0), k, features.size(1))
    if check_loop_indices:
        torch.testing.assert_close(gathered_features, torch.cat(loop_feature_gathers))


def _assert_ragged_true_tie_neighbors_are_valid(
    points,
    point_counts,
    queries,
    query_counts,
    k,
    *,
    expected_distance_sq,
):
    batch_offsets = _offsets(point_counts)
    query_offsets = _offsets(query_counts)
    bvh = torchbvh.build_bvh_ragged(points.contiguous(), batch_offsets)
    indices, distances, positions = torchbvh.query_knn_ragged(
        bvh,
        queries.contiguous(),
        query_offsets,
        k,
        source_points=points.contiguous(),
    )

    assert indices.shape == (queries.size(0), k)
    assert distances.shape == (queries.size(0), k)
    assert positions.shape == (queries.size(0), k, points.size(1))
    assert torch.all(distances[:, :-1] <= distances[:, 1:])
    torch.testing.assert_close(
        distances,
        (positions - queries[:, None, :]).square().sum(dim=-1),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        distances,
        torch.full_like(distances, expected_distance_sq),
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    query_start = 0
    for batch, (point_count, query_count) in enumerate(zip(point_counts, query_counts)):
        p0, p1 = int(batch_offsets[batch]), int(batch_offsets[batch + 1])
        sample_points = points[p0:p1]
        sample_queries = queries[query_start : query_start + query_count]
        sample_indices = indices[query_start : query_start + query_count]
        sample_positions = positions[query_start : query_start + query_count]
        all_distances = (sample_queries[:, None, :] - sample_points[None, :, :]).square().sum(dim=-1)
        candidate_mask = torch.isclose(
            all_distances,
            torch.full_like(all_distances, expected_distance_sq),
            rtol=1.0e-6,
            atol=1.0e-6,
        )

        assert int(sample_indices.min()) >= 0
        assert int(sample_indices.max()) < point_count
        torch.testing.assert_close(sample_positions, sample_points[sample_indices])
        assert torch.all(torch.gather(candidate_mask, 1, sample_indices))
        query_start += query_count


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("k", [4, 8, 16])
def test_ragged_knn_matches_single_sample_loop_for_uneven_random_clouds(dim, k):
    assert torch.cuda.is_available()
    torch.manual_seed(14100 + dim * 10 + k)
    point_counts = [k, k + 3, k + 9]
    query_counts = [3, 6, 4]
    points = torch.cat(
        [
            torch.rand((count, dim), device="cuda", dtype=torch.float32) + batch * 20.0
            for batch, count in enumerate(point_counts)
        ],
        dim=0,
    )
    queries = torch.cat(
        [
            torch.rand((count, dim), device="cuda", dtype=torch.float32) + batch * 20.0
            for batch, count in enumerate(query_counts)
        ],
        dim=0,
    )

    _assert_ragged_matches_single_loop(points, point_counts, queries, query_counts, k)


def test_ragged_knn_handles_duplicates_and_uneven_pathological_clouds():
    assert torch.cuda.is_available()
    base9 = torch.linspace(-1.0, 1.0, 9, device="cuda", dtype=torch.float32)
    duplicate = torch.zeros((9, 2), device="cuda", dtype=torch.float32)
    line = torch.stack((base9, -2.0 * base9), dim=1)
    base13 = torch.linspace(-2.0, 2.0, 13, device="cuda", dtype=torch.float32)
    curve = torch.stack((base13, base13.square()), dim=1)
    points = torch.cat((duplicate, line, curve), dim=0)
    queries = torch.cat(
        (
            duplicate[:4] + torch.tensor([0.01, -0.02], device="cuda"),
            line[:5] + torch.tensor([-0.02, 0.03], device="cuda"),
            curve[:6] + torch.tensor([0.04, 0.01], device="cuda"),
        ),
        dim=0,
    )

    _assert_ragged_matches_single_loop(points, [9, 9, 13], queries, [4, 5, 6], 4)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("k", [4, 8, 16])
def test_ragged_knn_robust_matrix_with_uneven_sizes_and_gathers(dim, k):
    assert torch.cuda.is_available()
    torch.manual_seed(14300 + dim * 100 + k)
    point_counts = [k + 5, k + 21, k + 43]
    query_counts = [5, 9, 13]

    duplicate = torch.full((point_counts[0], dim), -1.25, device="cuda", dtype=torch.float32)

    random_with_ties = torch.rand((point_counts[1], dim), device="cuda", dtype=torch.float32) - 0.5
    random_with_ties[: k + 2] = 0.0
    random_with_ties[-2:, 0] = torch.tensor([75.0, -75.0], device="cuda")
    if dim == 3:
        random_with_ties[-2:, 1:] = torch.tensor([[12.0, -6.0], [-12.0, 6.0]], device="cuda")
    else:
        random_with_ties[-2:, 1] = torch.tensor([12.0, -12.0], device="cuda")
    random_with_ties += 50.0

    base = torch.linspace(-2.0, 2.0, point_counts[2], device="cuda", dtype=torch.float32)
    if dim == 2:
        structured = torch.stack((base, base.sin()), dim=1)
    else:
        structured = torch.stack((base, base.sin(), base.cos()), dim=1)
    structured -= 50.0

    points = torch.cat((duplicate, random_with_ties, structured), dim=0)
    displacement = torch.linspace(0.01, 0.04, dim, device="cuda", dtype=torch.float32)
    queries = torch.cat(
        (
            duplicate[: query_counts[0]],
            torch.cat(
                (
                    random_with_ties[:4],
                    random_with_ties[4:7] + displacement,
                    random_with_ties[-2:] + 0.2 * displacement,
                ),
                dim=0,
            ),
            torch.cat(
                (
                    structured[:5],
                    structured[5:11] + displacement,
                    torch.rand((2, dim), device="cuda", dtype=torch.float32) - 50.0,
                ),
                dim=0,
            ),
        ),
        dim=0,
    )

    _assert_ragged_matches_single_loop(
        points,
        point_counts,
        queries,
        query_counts,
        k,
        check_loop_indices=False,
    )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("k", [4, 8, 16])
def test_ragged_knn_true_ties_use_valid_local_members_without_stable_order(dim, k):
    assert torch.cuda.is_available()
    if dim == 2:
        directions = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
            device="cuda",
            dtype=torch.float32,
        )
    else:
        directions = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            device="cuda",
            dtype=torch.float32,
        )
    point_counts = [k + 3, k + 7, k + 11]
    query_counts = [2, 3, 1]
    shifts = [
        torch.full((dim,), -40.0, device="cuda", dtype=torch.float32),
        torch.full((dim,), 40.0, device="cuda", dtype=torch.float32),
        torch.linspace(-25.0, 25.0, dim, device="cuda", dtype=torch.float32),
    ]
    duplicate_point = torch.zeros(dim, device="cuda", dtype=torch.float32)
    duplicate_point[0] = 1.0
    alternate_duplicate_point = torch.zeros(dim, device="cuda", dtype=torch.float32)
    alternate_duplicate_point[-1] = -1.0
    samples = [
        duplicate_point.expand(point_counts[0], dim) + shifts[0],
        directions.repeat((point_counts[1] + directions.size(0) - 1) // directions.size(0), 1)[
            : point_counts[1]
        ]
        + shifts[1],
        alternate_duplicate_point.expand(point_counts[2], dim) + shifts[2],
    ]
    points = torch.cat(samples, dim=0)
    queries = torch.cat(
        [shift.expand(query_count, dim) for shift, query_count in zip(shifts, query_counts)],
        dim=0,
    )

    _assert_ragged_true_tie_neighbors_are_valid(
        points,
        point_counts,
        queries,
        query_counts,
        k,
        expected_distance_sq=1.0,
    )


def test_ragged_knn_indices_are_local_not_global_packed():
    assert torch.cuda.is_available()
    torch.manual_seed(14200)
    point_counts = [8, 11, 14]
    query_counts = [3, 3, 3]
    points = torch.cat(
        [
            torch.rand((count, 3), device="cuda", dtype=torch.float32) + batch * 100.0
            for batch, count in enumerate(point_counts)
        ],
        dim=0,
    )
    queries = torch.cat(
        [
            points[sum(point_counts[:batch]) : sum(point_counts[:batch]) + count].contiguous()
            for batch, count in enumerate(query_counts)
        ],
        dim=0,
    )
    bvh = torchbvh.build_bvh_ragged(points.contiguous(), _offsets(point_counts))
    indices, distances = torchbvh.query_knn_ragged(
        bvh,
        queries.contiguous(),
        _offsets(query_counts),
        4,
    )

    start = 0
    for count, q_count in zip(point_counts, query_counts):
        sample_indices = indices[start : start + q_count]
        assert int(sample_indices.min()) >= 0
        assert int(sample_indices.max()) < count
        start += q_count
    assert torch.all(distances[:, 0] == 0.0)


def test_query_knn_ragged_neighbor_positions_gradients_flow_to_source_points_only():
    assert torch.cuda.is_available()
    torch.manual_seed(51030)
    point_counts = [8, 10, 12]
    query_counts = [3, 4, 5]
    points = torch.cat(
        [
            torch.randn((count, 3), device="cuda", dtype=torch.float32) + batch * 20.0
            for batch, count in enumerate(point_counts)
        ],
        dim=0,
    ).contiguous().requires_grad_()
    queries = torch.cat(
        [
            points.detach()[sum(point_counts[:batch]) : sum(point_counts[:batch]) + query_count] + 0.025
            for batch, query_count in enumerate(query_counts)
        ],
        dim=0,
    ).contiguous().requires_grad_()
    bvh = torchbvh.build_bvh_ragged(points, _offsets(point_counts))

    indices, distances, neighbor_positions = torchbvh.query_knn_ragged(
        bvh,
        queries,
        _offsets(query_counts),
        4,
        source_points=points,
    )

    assert not indices.requires_grad
    assert not distances.requires_grad
    assert neighbor_positions.requires_grad
    neighbor_positions.square().sum().backward()
    assert points.grad is not None
    assert torch.count_nonzero(points.grad) > 0
    assert queries.grad is None


def test_ragged_knn_rejects_unsupported_k():
    assert torch.cuda.is_available()
    points = torch.rand((16, 2), device="cuda", dtype=torch.float32).contiguous()
    offsets = torch.tensor([0, 8, 16], device="cuda", dtype=torch.int64)
    bvh = torchbvh.build_bvh_ragged(points, offsets)

    with pytest.raises(ValueError, match="k must be 4, 8, or 16"):
        torchbvh.query_knn_ragged(bvh, points, offsets, 5)


def test_ragged_knn_rejects_invalid_handle_type():
    assert torch.cuda.is_available()
    points = torch.rand((8, 2), device="cuda", dtype=torch.float32).contiguous()
    single_bvh = torchbvh.build_bvh(points)

    with pytest.raises(TypeError, match="RaggedBVHHandle"):
        torchbvh.query_knn_ragged(
            single_bvh,
            points,
            torch.tensor([0, 8], device="cuda", dtype=torch.int64),
            4,
        )
