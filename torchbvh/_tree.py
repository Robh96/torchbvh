import torch

from . import _C


def smoke_add_one(input: torch.Tensor) -> torch.Tensor:
    """Return a CUDA tensor with one added to every element."""
    return _C.smoke_add_one(input)




def implicit_tree_summary(t: int) -> dict:
    """Return compiled implicit-tree arithmetic values for ``t`` real leaves."""
    return _C.implicit_tree_summary(t)


def implicit_tree_descendant(implicit_idx: int, levels_down: int, offset: int) -> int:
    """Return the offset-th descendant ``levels_down`` levels below a BFS node."""
    return _C.implicit_tree_descendant(implicit_idx, levels_down, offset)


def implicit_tree_ancestor(implicit_idx: int, levels_up: int, descendant_offset: int) -> int:
    """Return the ancestor that produced ``implicit_idx`` with ``descendant_offset``."""
    return _C.implicit_tree_ancestor(implicit_idx, levels_up, descendant_offset)


def morton_split2(x: int) -> int:
    """Return the 16-bit 2D Morton split for ``x``."""
    return _C.morton_split2(x)


def morton_split3(x: int) -> int:
    """Return the 10-bit 3D Morton split for ``x``."""
    return _C.morton_split3(x)


def morton_encode_2d(
    x: float,
    y: float,
    scene_min: tuple[float, float],
    scene_max: tuple[float, float],
) -> int:
    """Return the 32-bit 2D Morton code for one point in one scene AABB."""
    return _C.morton_encode_2d(x, y, scene_min[0], scene_min[1], scene_max[0], scene_max[1])


def morton_encode_3d(
    x: float,
    y: float,
    z: float,
    scene_min: tuple[float, float, float],
    scene_max: tuple[float, float, float],
) -> int:
    """Return the 32-bit 3D Morton code for one point in one scene AABB."""
    return _C.morton_encode_3d(
        x,
        y,
        z,
        scene_min[0],
        scene_min[1],
        scene_min[2],
        scene_max[0],
        scene_max[1],
        scene_max[2],
    )
