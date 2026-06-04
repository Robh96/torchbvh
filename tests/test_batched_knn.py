import pytest
import torch

import torchbvh


def _brute_force_batched(points: torch.Tensor, query_points: torch.Tensor, k: int):
    distances_sq = (query_points[:, :, None, :] - points[:, None, :, :]).square().sum(dim=-1)
    return distances_sq.topk(k, largest=False)


def _assert_batched_knn_matches(
    points: torch.Tensor,
    query_points: torch.Tensor,
    k: int,
    *,
    check_loop_indices: bool = True,
):
    bvh = torchbvh.build_bvh_batched(points.contiguous())
    indices, distances, neighbor_positions = torchbvh.query_knn_batched(
        bvh,
        query_points.contiguous(),
        k,
        source_points=points.contiguous(),
    )
    expected_distances, _ = _brute_force_batched(points, query_points, k)

    assert indices.shape == (*query_points.shape[:2], k)
    assert distances.shape == (*query_points.shape[:2], k)
    assert neighbor_positions.shape == (*query_points.shape[:2], k, points.size(-1))
    assert indices.dtype == torch.int64
    assert distances.dtype == torch.float32
    assert torch.all(indices >= 0)
    assert torch.all(indices < points.size(1))
    assert torch.all(distances[:, :, :-1] <= distances[:, :, 1:])

    actual_distances = (neighbor_positions - query_points[:, :, None, :]).square().sum(dim=-1)
    expected_positions = torch.gather(
        points[:, None, :, :].expand(points.size(0), query_points.size(1), points.size(1), points.size(2)),
        2,
        indices.unsqueeze(-1).expand(-1, -1, -1, points.size(2)),
    )
    torch.testing.assert_close(neighbor_positions, expected_positions)
    torch.testing.assert_close(distances, actual_distances, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(distances, expected_distances, rtol=1.0e-5, atol=1.0e-5)

    features = torch.arange(
        points.size(0) * points.size(1) * 5,
        device=points.device,
        dtype=torch.float32,
    ).reshape(points.size(0), points.size(1), 5)
    gathered_features = torch.gather(
        features[:, None, :, :].expand(points.size(0), query_points.size(1), points.size(1), 5),
        2,
        indices.unsqueeze(-1).expand(-1, -1, -1, 5),
    )
    assert gathered_features.shape == (points.size(0), query_points.size(1), k, 5)

    loop_indices = []
    loop_distances = []
    for batch in range(points.size(0)):
        single = torchbvh.build_bvh(points[batch].contiguous())
        single_indices, single_distances = torchbvh.query_knn(
            single,
            query_points[batch].contiguous(),
            k,
        )
        loop_indices.append(single_indices)
        loop_distances.append(single_distances)
    if check_loop_indices:
        torch.testing.assert_close(indices, torch.stack(loop_indices), rtol=0, atol=0)
    torch.testing.assert_close(distances, torch.stack(loop_distances), rtol=1.0e-5, atol=1.0e-5)


def _assert_batched_true_tie_neighbors_are_valid(
    points: torch.Tensor,
    query_points: torch.Tensor,
    k: int,
    *,
    expected_distance_sq: float,
):
    bvh = torchbvh.build_bvh_batched(points.contiguous())
    indices, distances, neighbor_positions = torchbvh.query_knn_batched(
        bvh,
        query_points.contiguous(),
        k,
        source_points=points.contiguous(),
    )
    all_distances = (query_points[:, :, None, :] - points[:, None, :, :]).square().sum(dim=-1)
    candidate_mask = torch.isclose(
        all_distances,
        torch.full_like(all_distances, expected_distance_sq),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    expected_positions = torch.gather(
        points[:, None, :, :].expand(points.size(0), query_points.size(1), points.size(1), points.size(2)),
        2,
        indices.unsqueeze(-1).expand(-1, -1, -1, points.size(2)),
    )

    assert indices.shape == (*query_points.shape[:2], k)
    assert distances.shape == (*query_points.shape[:2], k)
    assert neighbor_positions.shape == (*query_points.shape[:2], k, points.size(-1))
    assert torch.all(indices >= 0)
    assert torch.all(indices < points.size(1))
    assert torch.all(distances[:, :, :-1] <= distances[:, :, 1:])
    torch.testing.assert_close(neighbor_positions, expected_positions)
    torch.testing.assert_close(
        distances,
        (neighbor_positions - query_points[:, :, None, :]).square().sum(dim=-1),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        distances,
        torch.full_like(distances, expected_distance_sq),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert torch.all(torch.gather(candidate_mask, 2, indices))


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("n", [16, 17, 31])
@pytest.mark.parametrize("k", [4, 8, 16])
def test_batched_knn_matches_brute_force_for_random_clouds(dim, n, k):
    assert torch.cuda.is_available()
    torch.manual_seed(13100 + dim * 100 + n * 10 + k)
    points = torch.rand((3, n, dim), device="cuda", dtype=torch.float32)
    query_points = torch.rand((3, 9, dim), device="cuda", dtype=torch.float32)

    _assert_batched_knn_matches(points, query_points, k)


@pytest.mark.parametrize("k", [4, 8])
def test_batched_knn_handles_pathological_fixed_size_clouds(k):
    assert torch.cuda.is_available()
    base17 = torch.linspace(-1.0, 1.0, 17, device="cuda", dtype=torch.float32)
    duplicate_2d = torch.zeros((17, 2), device="cuda", dtype=torch.float32)
    collinear_2d = torch.stack((base17, -2.0 * base17), dim=1)
    shifted_2d = torch.stack((base17.square(), torch.full_like(base17, 3.0)), dim=1)
    points_2d = torch.stack((duplicate_2d, collinear_2d, shifted_2d), dim=0)
    queries_2d = points_2d[:, :7, :] + torch.tensor([0.03, -0.02], device="cuda")
    _assert_batched_knn_matches(points_2d, queries_2d, k, check_loop_indices=False)

    base18 = torch.linspace(-2.0, 2.0, 18, device="cuda", dtype=torch.float32)
    duplicate_3d = torch.full((18, 3), 1.25, device="cuda", dtype=torch.float32)
    collinear_3d = torch.stack((base18, 0.5 * base18, -base18), dim=1)
    planar_3d = torch.stack((base18, base18.square(), torch.zeros_like(base18)), dim=1)
    points_3d = torch.stack((duplicate_3d, collinear_3d, planar_3d), dim=0)
    queries_3d = points_3d[:, :6, :] + torch.tensor([0.04, -0.01, 0.02], device="cuda")
    _assert_batched_knn_matches(points_3d, queries_3d, k, check_loop_indices=False)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("k", [4, 8, 16])
def test_batched_knn_robust_matrix_has_no_cross_sample_leakage(dim, k):
    assert torch.cuda.is_available()
    torch.manual_seed(13300 + dim * 100 + k)
    n = 80
    all_duplicate = torch.full((n, dim), -2.0, device="cuda", dtype=torch.float32)

    random_with_ties = torch.rand((n, dim), device="cuda", dtype=torch.float32) * 2.0 - 1.0
    random_with_ties[: k + 3] = 0.0
    random_with_ties[k + 3 : k + 5] = 0.5
    random_with_ties[-2:, 0] = torch.tensor([40.0, -40.0], device="cuda")
    if dim == 3:
        random_with_ties[-2:, 1:] = torch.tensor([[20.0, -5.0], [-20.0, 5.0]], device="cuda")
    else:
        random_with_ties[-2:, 1] = torch.tensor([20.0, -20.0], device="cuda")

    base = torch.linspace(-1.5, 1.5, n, device="cuda", dtype=torch.float32)
    if dim == 2:
        structured = torch.stack((base, base.square()), dim=1)
    else:
        structured = torch.stack((base, base.square(), torch.zeros_like(base)), dim=1)

    shifts = torch.stack(
        (
            torch.zeros(dim, device="cuda", dtype=torch.float32),
            torch.full((dim,), 100.0, device="cuda", dtype=torch.float32),
            torch.full((dim,), -100.0, device="cuda", dtype=torch.float32),
        ),
        dim=0,
    )
    points = torch.stack((all_duplicate, random_with_ties, structured), dim=0) + shifts[:, None, :]
    displacement = torch.linspace(0.01, 0.03, dim, device="cuda", dtype=torch.float32)
    random_queries = torch.rand((3, 11, dim), device="cuda", dtype=torch.float32) * 2.0 - 1.0
    query_points = torch.cat(
        (
            points[:, :17, :],
            points[:, 17:34, :] + displacement,
            points[:, -2:, :] + 0.25 * displacement,
            random_queries + shifts[:, None, :],
        ),
        dim=1,
    )

    _assert_batched_knn_matches(points, query_points, k, check_loop_indices=False)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("k", [4, 8, 16])
def test_batched_knn_true_ties_are_local_members_without_stable_order(dim, k):
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
    n = 24
    symmetric = directions.repeat((n + directions.size(0) - 1) // directions.size(0), 1)[:n]
    duplicate_point = torch.zeros(dim, device="cuda", dtype=torch.float32)
    duplicate_point[0] = 1.0
    alternate_duplicate_point = torch.zeros(dim, device="cuda", dtype=torch.float32)
    alternate_duplicate_point[-1] = -1.0
    all_duplicate = duplicate_point.expand(n, dim)
    shifted_duplicate = alternate_duplicate_point.expand(n, dim)
    shifts = torch.stack(
        (
            torch.full((dim,), -100.0, device="cuda", dtype=torch.float32),
            torch.full((dim,), 100.0, device="cuda", dtype=torch.float32),
            torch.linspace(-75.0, 75.0, dim, device="cuda", dtype=torch.float32),
        )
    )
    points = torch.stack((symmetric, all_duplicate, shifted_duplicate), dim=0) + shifts[:, None, :]
    query_points = torch.stack((shifts[0], shifts[1], shifts[2]), dim=0)[:, None, :].expand(3, 2, dim)

    _assert_batched_true_tie_neighbors_are_valid(points, query_points, k, expected_distance_sq=1.0)


def test_batched_knn_indices_are_local_not_flattened():
    assert torch.cuda.is_available()
    torch.manual_seed(13200)
    points = torch.rand((4, 20, 2), device="cuda", dtype=torch.float32)
    points[1:] += torch.tensor([10.0, -7.0], device="cuda")
    bvh = torchbvh.build_bvh_batched(points.contiguous())
    indices, distances = torchbvh.query_knn_batched(bvh, points[:, :5, :].contiguous(), 4)

    assert int(indices.max()) < points.size(1)
    assert int(indices.min()) >= 0
    assert torch.all(distances[:, :, 0] == 0.0)


def test_batched_knn_point_warp_multihead_smoke_reuses_bvh_and_gathers_values():
    assert torch.cuda.is_available()
    torch.manual_seed(13250)
    batch_size, n, heads, dim, k, channels = 2, 32, 4, 3, 4, 5
    points = torch.rand((batch_size, n, dim), device="cuda", dtype=torch.float32)
    points[1] += torch.tensor([4.0, -3.0, 2.0], device="cuda")
    displacements = 0.025 * torch.randn((batch_size, n, heads, dim), device="cuda")
    query_points = (points[:, :, None, :] + displacements).contiguous()
    flat_queries = query_points.reshape(batch_size, n * heads, dim).contiguous()
    values = torch.randn((batch_size, n, heads, channels), device="cuda", dtype=torch.float32)
    values[1] += 100.0

    bvh = torchbvh.build_bvh_batched(points.contiguous())
    indices, distances, neighbor_positions = torchbvh.query_knn_batched(
        bvh,
        flat_queries,
        k,
        source_points=points.contiguous(),
    )

    expected_distances, _ = _brute_force_batched(points, flat_queries, k)
    assert bvh["batch_size"] == batch_size
    assert indices.shape == (batch_size, n * heads, k)
    assert distances.shape == (batch_size, n * heads, k)
    assert neighbor_positions.shape == (batch_size, n * heads, k, dim)
    assert torch.all(indices >= 0)
    assert torch.all(indices < n)
    assert torch.all(distances[:, :, :-1] <= distances[:, :, 1:])
    torch.testing.assert_close(distances, expected_distances, rtol=1.0e-5, atol=2.0e-5)
    expected_positions = torch.stack([points[b, indices[b]] for b in range(batch_size)])
    torch.testing.assert_close(neighbor_positions, expected_positions)

    indices_by_query_head = indices.reshape(batch_size, n, heads, k)
    source_values_by_query_head = values[:, None, :, :, :].expand(batch_size, n, n, heads, channels)
    source_values_by_query_head = source_values_by_query_head.permute(0, 1, 3, 2, 4).contiguous()
    gather_index = indices_by_query_head.unsqueeze(-1).expand(batch_size, n, heads, k, channels)
    neighbor_values = torch.gather(source_values_by_query_head, 3, gather_index)

    expected_values = torch.empty_like(neighbor_values)
    for batch in range(batch_size):
        for point in range(n):
            for head in range(heads):
                expected_values[batch, point, head] = values[
                    batch,
                    indices_by_query_head[batch, point, head],
                    head,
                ]
    torch.testing.assert_close(neighbor_values, expected_values)


def test_batched_knn_rejects_unsupported_k():
    assert torch.cuda.is_available()
    points = torch.rand((2, 8, 2), device="cuda", dtype=torch.float32).contiguous()
    bvh = torchbvh.build_bvh_batched(points)

    with pytest.raises(ValueError, match="k must be 4, 8, or 16"):
        torchbvh.query_knn_batched(bvh, points, 5)


def test_batched_knn_rejects_invalid_handle_type():
    assert torch.cuda.is_available()
    points = torch.rand((2, 8, 2), device="cuda", dtype=torch.float32).contiguous()
    single_bvh = torchbvh.build_bvh(points[0].contiguous())

    with pytest.raises(TypeError, match="BatchedBVHHandle"):
        torchbvh.query_knn_batched(single_bvh, points, 4)
