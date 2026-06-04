import pytest
import torch

import torchbvh


def _brute_force_knn(points: torch.Tensor, query_points: torch.Tensor, k: int):
    distances_sq = (query_points[:, None, :] - points[None, :, :]).square().sum(dim=2)
    return distances_sq.topk(k, largest=False)


def _assert_knn_matches(points: torch.Tensor, query_points: torch.Tensor, k: int, check_indices: bool):
    bvh = torchbvh.build_bvh(points.contiguous())
    indices, distances = torchbvh.query_knn(bvh, query_points.contiguous(), k)
    expected_distances, expected_indices = _brute_force_knn(points, query_points, k)

    assert indices.shape == (query_points.shape[0], k)
    assert distances.shape == (query_points.shape[0], k)
    assert indices.dtype == torch.int64
    assert distances.dtype == torch.float32
    assert torch.all(indices >= 0)
    assert torch.all(indices < points.shape[0])
    assert torch.all(distances[:, :-1] <= distances[:, 1:])

    gathered = points[indices]
    actual_distances = (gathered - query_points[:, None, :]).square().sum(dim=2)
    torch.testing.assert_close(distances, actual_distances)
    torch.testing.assert_close(distances, expected_distances, rtol=1.0e-5, atol=1.0e-5)
    if check_indices:
        torch.testing.assert_close(indices, expected_indices)


def _assert_knn_matches_with_gathers(
    points: torch.Tensor,
    query_points: torch.Tensor,
    k: int,
    *,
    check_indices: bool,
):
    bvh = torchbvh.build_bvh(points.contiguous())
    indices, distances, neighbor_positions = torchbvh.query_knn(
        bvh,
        query_points.contiguous(),
        k,
        source_points=points.contiguous(),
    )
    expected_distances, expected_indices = _brute_force_knn(points, query_points, k)

    assert indices.shape == (query_points.shape[0], k)
    assert distances.shape == (query_points.shape[0], k)
    assert neighbor_positions.shape == (query_points.shape[0], k, points.shape[1])
    assert torch.all(indices >= 0)
    assert torch.all(indices < points.shape[0])
    assert torch.all(distances[:, :-1] <= distances[:, 1:])

    torch.testing.assert_close(neighbor_positions, points[indices])
    torch.testing.assert_close(
        distances,
        (neighbor_positions - query_points[:, None, :]).square().sum(dim=2),
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    torch.testing.assert_close(distances, expected_distances, rtol=1.0e-5, atol=1.0e-5)
    if check_indices:
        torch.testing.assert_close(indices, expected_indices)

    features = torch.arange(
        points.shape[0] * 5,
        device=points.device,
        dtype=torch.float32,
    ).reshape(points.shape[0], 5)
    gathered_features = features[indices]
    assert gathered_features.shape == (query_points.shape[0], k, features.shape[1])
    torch.testing.assert_close(gathered_features, features[indices])


def _assert_true_tie_neighbors_are_valid(
    points: torch.Tensor,
    query_points: torch.Tensor,
    k: int,
    *,
    expected_distance_sq: float,
):
    bvh = torchbvh.build_bvh(points.contiguous())
    indices, distances, neighbor_positions = torchbvh.query_knn(
        bvh,
        query_points.contiguous(),
        k,
        source_points=points.contiguous(),
    )
    all_distances = (query_points[:, None, :] - points[None, :, :]).square().sum(dim=2)
    candidate_mask = torch.isclose(
        all_distances,
        torch.full_like(all_distances, expected_distance_sq),
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    assert indices.shape == (query_points.shape[0], k)
    assert distances.shape == (query_points.shape[0], k)
    assert neighbor_positions.shape == (query_points.shape[0], k, points.shape[1])
    assert torch.all(indices >= 0)
    assert torch.all(indices < points.shape[0])
    assert torch.all(distances[:, :-1] <= distances[:, 1:])
    torch.testing.assert_close(neighbor_positions, points[indices])
    torch.testing.assert_close(
        distances,
        (neighbor_positions - query_points[:, None, :]).square().sum(dim=2),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        distances,
        torch.full_like(distances, expected_distance_sq),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert torch.all(torch.gather(candidate_mask, 1, indices))


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("k", [4, 8, 16])
def test_knn_matches_brute_force_for_uniform_random_clouds(dim, k):
    assert torch.cuda.is_available()
    torch.manual_seed(1000 + dim * 10 + k)
    points = torch.rand((64, dim), device="cuda", dtype=torch.float32)
    query_points = torch.rand((23, dim), device="cuda", dtype=torch.float32)

    _assert_knn_matches(points, query_points, k, check_indices=True)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("k", [4, 8, 16])
def test_knn_matches_brute_force_for_self_query(dim, k):
    assert torch.cuda.is_available()
    torch.manual_seed(2000 + dim * 10 + k)
    points = torch.rand((48, dim), device="cuda", dtype=torch.float32)

    _assert_knn_matches(points, points, k, check_indices=True)
    bvh = torchbvh.build_bvh(points.contiguous())
    indices, distances = torchbvh.query_knn(bvh, points.contiguous(), k)
    assert torch.all(distances[:, 0] == 0.0)
    torch.testing.assert_close(indices[:, 0], torch.arange(points.shape[0], device="cuda"))


@pytest.mark.parametrize("k", [4, 8])
def test_knn_matches_brute_force_for_displaced_queries(k):
    assert torch.cuda.is_available()
    base = torch.linspace(-2.0, 2.0, 32, device="cuda", dtype=torch.float32)
    points = torch.stack((base, base.sin(), base.cos()), dim=1)
    displacement = torch.tensor([0.07, -0.11, 0.05], device="cuda", dtype=torch.float32)
    query_points = points + displacement

    _assert_knn_matches(points, query_points, k, check_indices=True)


@pytest.mark.parametrize(
    "points,k",
    [
        (
            torch.stack(
                (
                    torch.linspace(-3.0, 3.0, 17, device="cuda"),
                    torch.linspace(6.0, -6.0, 17, device="cuda"),
                ),
                dim=1,
            ),
            4,
        ),
        (
            torch.stack(
                (
                    torch.linspace(-2.0, 2.0, 19, device="cuda"),
                    torch.linspace(4.0, -4.0, 19, device="cuda"),
                    torch.linspace(1.0, 9.0, 19, device="cuda"),
                ),
                dim=1,
            ),
            8,
        ),
        (
            torch.tensor(
                [
                    [-2.0, -1.0, 7.0],
                    [-1.0, 2.0, 7.0],
                    [0.0, -3.0, 7.0],
                    [1.0, 4.0, 7.0],
                    [2.0, -5.0, 7.0],
                    [3.0, 6.0, 7.0],
                    [4.0, -7.0, 7.0],
                    [5.0, 8.0, 7.0],
                    [6.0, -9.0, 7.0],
                    [7.0, 10.0, 7.0],
                    [8.0, -11.0, 7.0],
                    [9.0, 12.0, 7.0],
                ],
                device="cuda",
                dtype=torch.float32,
            ),
            4,
        ),
        (torch.full((18, 3), 2.5, device="cuda", dtype=torch.float32), 16),
        (
            torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0e6, 1.0e-3],
                    [2.0e6, -1.0e-3],
                    [3.0e6, 2.0e-3],
                    [4.0e6, -2.0e-3],
                    [5.0e6, 3.0e-3],
                    [6.0e6, -3.0e-3],
                    [7.0e6, 4.0e-3],
                    [8.0e6, -4.0e-3],
                ],
                device="cuda",
                dtype=torch.float32,
            ),
            8,
        ),
        (
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0e-3, 1.0e6, 2.0],
                    [-1.0e-3, 2.0e6, -2.0],
                    [2.0e-3, 3.0e6, 4.0],
                    [-2.0e-3, 4.0e6, -4.0],
                    [3.0e-3, 5.0e6, 6.0],
                    [-3.0e-3, 6.0e6, -6.0],
                    [4.0e-3, 7.0e6, 8.0],
                    [-4.0e-3, 8.0e6, -8.0],
                ],
                device="cuda",
                dtype=torch.float32,
            ),
            8,
        ),
        (
            torch.cat(
                (
                    torch.rand((20, 2), device="cuda", dtype=torch.float32),
                    torch.tensor([[1000.0, -1000.0]], device="cuda", dtype=torch.float32),
                ),
                dim=0,
            ),
            4,
        ),
    ],
)
def test_knn_matches_brute_force_for_pathological_clouds(points, k):
    assert torch.cuda.is_available()
    query_points = points[: min(7, points.shape[0])] + 0.125

    _assert_knn_matches(points, query_points, k, check_indices=False)


def test_knn_handles_duplicate_points_without_requiring_tie_order():
    assert torch.cuda.is_available()
    points = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    query_points = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device="cuda", dtype=torch.float32)

    _assert_knn_matches(points, query_points, 4, check_indices=False)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("k", [4, 8, 16])
def test_knn_true_tie_all_duplicate_membership_policy(dim, k):
    assert torch.cuda.is_available()
    points = torch.full((20, dim), 3.5, device="cuda", dtype=torch.float32)
    query_points = torch.full((3, dim), 3.5, device="cuda", dtype=torch.float32)

    _assert_true_tie_neighbors_are_valid(points, query_points, k, expected_distance_sq=0.0)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("k", [4, 8, 16])
def test_knn_true_tie_symmetric_candidates_do_not_require_order(dim, k):
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
    points = torch.cat(
        (
            directions.repeat_interleave(4, dim=0),
            3.0 * directions,
        ),
        dim=0,
    )
    query_points = torch.zeros((2, dim), device="cuda", dtype=torch.float32)

    _assert_true_tie_neighbors_are_valid(points, query_points, k, expected_distance_sq=1.0)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("k", [4, 8, 16])
def test_knn_robust_matrix_with_source_position_and_feature_gathers(dim, k):
    assert torch.cuda.is_available()
    torch.manual_seed(2400 + dim * 100 + k)
    random_points = torch.rand((96, dim), device="cuda", dtype=torch.float32) * 2.0 - 1.0
    tie_block = torch.zeros((k + 2, dim), device="cuda", dtype=torch.float32)
    duplicate_pair = torch.full((2, dim), 0.25, device="cuda", dtype=torch.float32)
    outliers = torch.full((2, dim), 0.0, device="cuda", dtype=torch.float32)
    outliers[0, 0] = 50.0
    outliers[1, 0] = -50.0
    if dim == 3:
        outliers[:, 1:] = torch.tensor([[30.0, -10.0], [-30.0, 10.0]], device="cuda")
    else:
        outliers[:, 1] = torch.tensor([30.0, -30.0], device="cuda")
    points = torch.cat((random_points, tie_block, duplicate_pair, outliers), dim=0)

    displacement = torch.linspace(0.015, 0.045, dim, device="cuda", dtype=torch.float32)
    query_points = torch.cat(
        (
            points[:19],
            points[19:38] + displacement,
            torch.zeros((5, dim), device="cuda", dtype=torch.float32),
            outliers + 0.1 * displacement,
            torch.rand((17, dim), device="cuda", dtype=torch.float32) * 2.0 - 1.0,
        ),
        dim=0,
    )

    _assert_knn_matches_with_gathers(points, query_points, k, check_indices=False)


def test_knn_rejects_unsupported_k():
    assert torch.cuda.is_available()
    points = torch.rand((8, 2), device="cuda", dtype=torch.float32)
    bvh = torchbvh.build_bvh(points.contiguous())

    with pytest.raises(ValueError, match="k must be 4, 8, or 16"):
        torchbvh.query_knn(bvh, points.contiguous(), 5)


def test_knn_rejects_invalid_handle_type():
    with pytest.raises(TypeError, match="BVHHandle or mapping"):
        torchbvh.query_knn(object(), torch.empty((4, 2)), 4)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("k", [4, 8, 16])
def test_knn_sorted_query_path_matches_unsorted(dim, k):
    assert torch.cuda.is_available()
    torch.manual_seed(3100 + dim * 10 + k)
    points = torch.rand((257, dim), device="cuda", dtype=torch.float32).contiguous()
    queries = torch.rand((123, dim), device="cuda", dtype=torch.float32).contiguous()
    bvh = torchbvh.build_bvh(points)

    idx_unsorted, dist_unsorted = torchbvh.query_knn(bvh, queries, k, sort_queries=False)
    idx_sorted, dist_sorted = torchbvh.query_knn(bvh, queries, k, sort_queries=True)
    idx_default, dist_default = torchbvh.query_knn(bvh, queries, k)

    torch.testing.assert_close(dist_sorted, dist_unsorted)
    torch.testing.assert_close(idx_sorted, idx_unsorted)
    torch.testing.assert_close(dist_default, dist_sorted)
    torch.testing.assert_close(idx_default, idx_sorted)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("k", [4, 8, 16])
def test_batched_knn_sorted_query_path_matches_unsorted(dim, k):
    assert torch.cuda.is_available()
    torch.manual_seed(3200 + dim * 10 + k)
    points = torch.rand((3, 257, dim), device="cuda", dtype=torch.float32).contiguous()
    queries = torch.rand((3, 123, dim), device="cuda", dtype=torch.float32).contiguous()
    bvh = torchbvh.build_bvh_batched(points)

    idx, dist = torchbvh.query_knn_batched(bvh, queries, k, sort_queries=False)
    idx_sorted, dist_sorted = torchbvh.query_knn_batched(
        bvh,
        queries,
        k,
        sort_queries=True,
    )
    idx_default, dist_default = torchbvh.query_knn_batched(bvh, queries, k)

    torch.testing.assert_close(dist_sorted, dist)
    torch.testing.assert_close(idx_sorted, idx)
    torch.testing.assert_close(dist_default, dist_sorted)
    torch.testing.assert_close(idx_default, idx_sorted)
