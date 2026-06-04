import pytest
import torch

import torchbvh


def _real_nodes(summary: dict) -> list[dict]:
    return [node for node in summary["nodes"] if not node["is_virtual"]]


def _node_by_implicit(summary: dict) -> dict[int, dict]:
    return {node["implicit_idx"]: node for node in summary["nodes"]}


def _real_memory_index(node: dict) -> int:
    assert not node["is_virtual"]
    assert node["memory_index"] is not None
    return node["memory_index"]


def _single_child_internal_count(summary: dict) -> int:
    by_implicit = _node_by_implicit(summary)
    count = 0
    for node in _real_nodes(summary):
        real_children = 0
        for child_idx in (2 * node["implicit_idx"] + 1, 2 * node["implicit_idx"] + 2):
            child = by_implicit.get(child_idx)
            if child is not None and not child["is_virtual"]:
                real_children += 1
        if real_children == 1:
            count += 1
    return count


def _assert_bvh_invariants(points: torch.Tensor):
    bvh = torchbvh.build_bvh(points.contiguous())
    node_aabbs = bvh["node_aabbs"]
    sorted_indices = bvh["sorted_indices"]
    summary = torchbvh.implicit_tree_summary(points.shape[0])
    by_implicit = _node_by_implicit(summary)
    leaf_level = bvh["leaf_level"]
    first_leaf = (1 << leaf_level) - 1
    dim = points.shape[1]

    assert bvh["num_leaves"] == points.shape[0]
    assert bvh["num_real_nodes"] == summary["real_node_count"]
    assert bvh["dim"] == dim
    assert bvh["virtual_leaves"] == summary["virtual_leaves"]
    assert node_aabbs.shape == (summary["real_node_count"], 2 * dim)
    assert sorted_indices.shape == (points.shape[0],)
    assert sorted_indices.dtype == torch.int64
    torch.testing.assert_close(
        torch.sort(sorted_indices).values,
        torch.arange(points.shape[0], device=points.device, dtype=torch.int64),
    )

    assert torch.isfinite(node_aabbs).all()
    assert torch.all(node_aabbs[:, :dim] <= node_aabbs[:, dim:])
    torch.testing.assert_close(bvh["scene_min"], points.min(dim=0).values)
    torch.testing.assert_close(bvh["scene_max"], points.max(dim=0).values)
    root_aabb = node_aabbs[_real_memory_index(by_implicit[0])]
    torch.testing.assert_close(root_aabb[:dim], bvh["scene_min"])
    torch.testing.assert_close(root_aabb[dim:], bvh["scene_max"])

    real_memory_indices = []
    for node in summary["nodes"]:
        if node["is_virtual"]:
            assert node["memory_index"] is None
        else:
            real_memory_indices.append(_real_memory_index(node))
    assert real_memory_indices == list(range(summary["real_node_count"]))

    leaf_aabbs = node_aabbs[
        [_real_memory_index(by_implicit[first_leaf + i]) for i in range(points.shape[0])]
    ]
    sorted_points = points[sorted_indices]
    torch.testing.assert_close(leaf_aabbs[:, :dim], sorted_points)
    torch.testing.assert_close(leaf_aabbs[:, dim:], sorted_points)

    for node in _real_nodes(summary):
        implicit_idx = node["implicit_idx"]
        if node["level"] == leaf_level:
            continue

        parent_aabb = node_aabbs[_real_memory_index(node)]
        real_children = []
        for child_idx in (2 * implicit_idx + 1, 2 * implicit_idx + 2):
            child = by_implicit[child_idx]
            if not child["is_virtual"]:
                real_children.append(child)

        assert real_children
        child_aabbs = node_aabbs[[_real_memory_index(child) for child in real_children]]
        assert torch.all(parent_aabb[:dim] <= child_aabbs[:, :dim])
        assert torch.all(parent_aabb[dim:] >= child_aabbs[:, dim:])

        if len(real_children) == 1:
            torch.testing.assert_close(parent_aabb, child_aabbs[0])

    return bvh


@pytest.mark.parametrize(
    "points",
    [
        torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.25, 0.75],
                [0.25, 0.75],
                [2.0, -1.0],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [1.0, 2.0, 0.0],
                [1.0, 2.0, 0.0],
                [3.0, 2.0, -1.0],
                [0.0, 2.0, 4.0],
                [5.0, 2.0, 4.0],
                [2.0, 2.0, 4.0],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
    ],
)
def test_bvh_build_validates_2d_and_3d_point_clouds(points):
    assert torch.cuda.is_available()
    _assert_bvh_invariants(points)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize(
    "n",
    [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65],
)
def test_bvh_build_validates_power_of_two_boundary_sizes(n, dim):
    assert torch.cuda.is_available()
    base = torch.arange(n, device="cuda", dtype=torch.float32)
    coords = []
    for d in range(dim):
        coords.append(((base * (d + 3)) % 11) + base * (0.125 + 0.25 * d))
    points = torch.stack(coords, dim=1)

    _assert_bvh_invariants(points)


@pytest.mark.parametrize(
    "points",
    [
        torch.tensor([[4.0, -2.0]], device="cuda", dtype=torch.float32),
        torch.tensor([[-1.0, 3.0, 5.0]], device="cuda", dtype=torch.float32),
        torch.tensor([[0.0, 0.0], [10.0, -2.0]], device="cuda", dtype=torch.float32),
        torch.tensor(
            [[0.0, 0.0, 0.0], [10.0, -2.0, 0.5]],
            device="cuda",
            dtype=torch.float32,
        ),
        torch.stack(
            (
                torch.linspace(-3.0, 3.0, 9, device="cuda"),
                torch.linspace(6.0, -6.0, 9, device="cuda"),
            ),
            dim=1,
        ),
        torch.stack(
            (
                torch.linspace(-2.0, 2.0, 11, device="cuda"),
                torch.linspace(4.0, -4.0, 11, device="cuda"),
                torch.linspace(1.0, 9.0, 11, device="cuda"),
            ),
            dim=1,
        ),
        torch.tensor(
            [
                [-2.0, -1.0, 7.0],
                [-1.0, 2.0, 7.0],
                [0.0, -3.0, 7.0],
                [1.0, 4.0, 7.0],
                [2.0, -5.0, 7.0],
                [3.0, 6.0, 7.0],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
        torch.full((9, 3), 2.5, device="cuda", dtype=torch.float32),
        torch.tensor(
            [
                [0.0, 0.0],
                [1.0e6, 1.0e-3],
                [2.0e6, -1.0e-3],
                [3.0e6, 2.0e-3],
                [4.0e6, -2.0e-3],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0e-3, 1.0e6, 2.0],
                [-1.0e-3, 2.0e6, -2.0],
                [2.0e-3, 3.0e6, 4.0],
                [-2.0e-3, 4.0e6, -4.0],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
    ],
)
def test_bvh_build_validates_pathological_point_clouds(points):
    assert torch.cuda.is_available()
    _assert_bvh_invariants(points)


@pytest.mark.parametrize(
    "points",
    [
        torch.tensor(
            [
                [1.0e-7, -2.0e-7],
                [1.5e-7, -2.5e-7],
                [2.0e-7, -1.5e-7],
                [2.5e-7, -3.0e-7],
                [3.0e-7, -1.0e-7],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [1.0, 1.0e-6, -1.0e3],
                [2.0, 1.5e-6, -5.0e2],
                [3.0, 2.0e-6, 0.0],
                [4.0, 2.5e-6, 5.0e2],
                [5.0, 3.0e-6, 1.0e3],
                [6.0, 3.5e-6, 1.5e3],
                [7.0, 4.0e-6, 2.0e3],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [1.0e-4, -1.0e8],
                [2.0e-4, -5.0e7],
                [3.0e-4, 0.0],
                [4.0e-4, 5.0e7],
                [5.0e-4, 1.0e8],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [-1.0e6, 1.0e-5, 2.0],
                [-5.0e5, 2.0e-5, 2.0],
                [0.0, 3.0e-5, 2.0],
                [5.0e5, 4.0e-5, 2.0],
                [1.0e6, 5.0e-5, 2.0],
                [1.5e6, 6.0e-5, 2.0],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
    ],
)
def test_bvh_build_validates_tiny_and_mixed_scale_axes(points):
    assert torch.cuda.is_available()
    _assert_bvh_invariants(points)


@pytest.mark.parametrize("n", [3, 5, 6, 9, 10, 17, 18])
def test_bvh_build_validates_single_child_internal_nodes_for_non_power_of_two_sizes(n):
    assert torch.cuda.is_available()
    points = torch.stack(
        (
            torch.arange(n, device="cuda", dtype=torch.float32),
            torch.arange(n, device="cuda", dtype=torch.float32).remainder(3),
            -torch.arange(n, device="cuda", dtype=torch.float32),
        ),
        dim=1,
    )
    summary = torchbvh.implicit_tree_summary(n)

    assert _single_child_internal_count(summary) > 0
    _assert_bvh_invariants(points)


def test_bvh_build_handles_fully_degenerate_2d_extent():
    assert torch.cuda.is_available()
    points = torch.tensor(
        [
            [7.0, -3.0],
            [7.0, -3.0],
            [7.0, -3.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )

    bvh = _assert_bvh_invariants(points)

    expected = points[0].expand_as(bvh["node_aabbs"][:, :2])
    torch.testing.assert_close(bvh["node_aabbs"][:, :2], expected)
    torch.testing.assert_close(bvh["node_aabbs"][:, 2:], bvh["node_aabbs"][:, :2])


def test_bvh_build_rejects_unsupported_inputs():
    assert torch.cuda.is_available()

    with pytest.raises(RuntimeError, match="D must be 2 or 3"):
        torchbvh.build_bvh(torch.zeros((4, 4), device="cuda", dtype=torch.float32))

    with pytest.raises(RuntimeError, match="float32"):
        torchbvh.build_bvh(torch.zeros((4, 2), device="cuda", dtype=torch.float64))
