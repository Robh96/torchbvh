import math

import torchbvh


TREE_SIZES = [1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 100, 1000, 1023, 1024, 1025]


def _leaf_level(t: int) -> int:
    return 0 if t <= 1 else math.ceil(math.log2(t))


def _virtual_leaves(t: int) -> int:
    return (1 << _leaf_level(t)) - t


def _popcount(x: int) -> int:
    return x.bit_count()


def _virtual_nodes_before_level(t: int, level: int) -> int:
    leaf = _leaf_level(t)
    lvl = _virtual_leaves(t) >> (leaf - level + 1)
    return 2 * lvl - _popcount(lvl)


def _real_nodes_at_level(t: int, level: int) -> int:
    leaf = _leaf_level(t)
    return (1 << level) - (_virtual_leaves(t) >> (leaf - level))


def _level_of(implicit_idx: int) -> int:
    return (implicit_idx + 1).bit_length() - 1


def _memory_index(t: int, implicit_idx: int) -> int:
    return implicit_idx - _virtual_nodes_before_level(t, _level_of(implicit_idx))


def _is_virtual(t: int, implicit_idx: int) -> bool:
    level = _level_of(implicit_idx)
    first = (1 << level) - 1
    return implicit_idx - first >= _real_nodes_at_level(t, level)


def _level_real_range(t: int, level: int) -> tuple[int, int]:
    first = (1 << level) - 1
    return first, first + _real_nodes_at_level(t, level) - 1


def test_implicit_tree_scalar_counts_and_ranges():
    for t in TREE_SIZES:
        summary = torchbvh.implicit_tree_summary(t)
        leaf = _leaf_level(t)

        assert summary["virtual_leaves"] == _virtual_leaves(t)
        assert summary["real_node_count"] == 2 * t - 1 + _popcount(_virtual_leaves(t))
        assert summary["leaf_level"] == leaf

        assert summary["virtual_nodes_at_level"] == [
            _virtual_nodes_before_level(t, level) for level in range(leaf + 1)
        ]
        assert summary["level_real_range"] == [
            _level_real_range(t, level) for level in range(leaf + 1)
        ]


def test_implicit_tree_memory_mapping_and_virtual_nodes():
    for t in TREE_SIZES:
        summary = torchbvh.implicit_tree_summary(t)
        real_memory_indices = []

        for node in summary["nodes"]:
            implicit_idx = node["implicit_idx"]
            assert node["level"] == _level_of(implicit_idx)
            assert node["is_virtual"] is _is_virtual(t, implicit_idx)

            if node["is_virtual"]:
                assert node["memory_index"] is None
            else:
                assert node["memory_index"] == _memory_index(t, implicit_idx)
                real_memory_indices.append(node["memory_index"])

        assert real_memory_indices == list(range(summary["real_node_count"]))


def test_implicit_tree_documented_t5_layout():
    summary = torchbvh.implicit_tree_summary(5)
    real_implicit = [
        node["implicit_idx"] for node in summary["nodes"] if not node["is_virtual"]
    ]
    virtual_implicit = [
        node["implicit_idx"] for node in summary["nodes"] if node["is_virtual"]
    ]

    assert summary["leaf_level"] == 3
    assert summary["virtual_leaves"] == 3
    assert summary["real_node_count"] == 11
    assert real_implicit == [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
    assert virtual_implicit == [6, 12, 13, 14]
    assert summary["level_real_range"] == [(0, 0), (1, 2), (3, 5), (7, 11)]


def test_implicit_tree_parent_and_child_helpers():
    for t in TREE_SIZES:
        for node in torchbvh.implicit_tree_summary(t)["nodes"]:
            i = node["implicit_idx"]
            assert node["left_child"] == 2 * i + 1
            assert node["right_child"] == 2 * i + 2
            if i == 0:
                assert node["parent"] is None
            else:
                assert node["parent"] == (i - 1) // 2


def test_implicit_tree_ancestor_descendant_helpers_are_inverse():
    for root in [0, 1, 2, 5, 16]:
        for levels_down in range(5):
            for offset in range(1 << levels_down):
                desc = torchbvh.implicit_tree_descendant(root, levels_down, offset)
                assert desc == ((root + 1) << levels_down) - 1 + offset
                anc = torchbvh.implicit_tree_ancestor(desc, levels_down, offset)
                assert anc == root
