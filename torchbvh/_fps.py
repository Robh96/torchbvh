from __future__ import annotations

from dataclasses import dataclass

import torch

from . import _C
from ._validation import _as_contiguous


@dataclass
class FPSResult:
    """Geometry and assignment metadata returned by FPS downsampling.

    ``nearest_anchor`` stores selection-order indices into ``indices``/``points``.
    If anchors are gathered in Morton order, remap with
    ``anchor_morton_idx = coarse_order[nearest_anchor[i]]``.
    """

    indices: torch.Tensor
    points: torch.Tensor
    nearest_anchor: torch.Tensor
    nearest_anchor_dist_sq: torch.Tensor
    anchor_radius: torch.Tensor
    anchor_counts: torch.Tensor
    coarse_order: torch.Tensor
    selection_order_indices: torch.Tensor


def _validate_points_and_m(
    prefix: str,
    points: torch.Tensor,
    M: int,
    seed: int,
    *,
    allow_batched: bool = False,
) -> None:
    if points.dim() not in (2, 3) or (points.dim() == 3 and not allow_batched):
        expected = "(N, D) or (B, N, D)" if allow_batched else "(N, D)"
        raise ValueError(f"{prefix}: points must have shape {expected}")
    dim_axis = points.dim() - 1
    n_axis = points.dim() - 2
    if points.size(dim_axis) not in (2, 3):
        raise ValueError(f"{prefix}: D must be 2 or 3")
    if points.dtype != torch.float32:
        raise ValueError(f"{prefix}: points must be float32")
    if not points.is_cuda:
        raise ValueError(f"{prefix}: points must be a CUDA tensor")
    if M < 1 or M > points.size(n_axis):
        raise ValueError(f"{prefix}: target token count must be in [1, N]")
    if seed < -1 or seed >= points.size(n_axis):
        raise ValueError(f"{prefix}: seed must be -1 or an original point index in [0, N)")


def _resolve_seed_batched(points: torch.Tensor, seed: int) -> torch.Tensor:
    if seed != -1:
        return torch.full((points.size(0),), int(seed), device=points.device, dtype=torch.int64)
    center = 0.5 * (points.min(dim=1).values + points.max(dim=1).values)
    return (points - center[:, None, :]).square().sum(dim=2).argmin(dim=1).to(torch.int64)


def _make_batched_result(
    points: torch.Tensor,
    fps_idx: torch.Tensor,
    nearest_anchor: torch.Tensor,
    nearest_anchor_dist_sq: torch.Tensor,
    anchor_radius: torch.Tensor,
    anchor_counts: torch.Tensor,
    fps_leaf_pos: torch.Tensor,
) -> FPSResult:
    B, _, D = points.shape
    M = int(fps_idx.size(1))
    coarse_order = torch.argsort(fps_leaf_pos, dim=1, stable=True)
    gathered_points = torch.gather(points, 1, fps_idx[:, :, None].expand(B, M, D))

    return FPSResult(
        indices=fps_idx,
        points=gathered_points,
        nearest_anchor=nearest_anchor.to(torch.int32),
        nearest_anchor_dist_sq=nearest_anchor_dist_sq,
        anchor_radius=anchor_radius,
        anchor_counts=anchor_counts,
        coarse_order=coarse_order,
        selection_order_indices=fps_idx,
    )


def _fps_exact_bucketed(
    points: torch.Tensor,
    M: int,
    seed: int = 0,
    *,
    bucket_size: int = 256,
    use_graph: bool = True,
    enable_pruning: bool = True,
) -> FPSResult:
    """Production exact bucketed FPS CUDA route.

    Anchor selection remains exact: one anchor is selected per round from exact
    per-bucket maxima. AABB pruning can skip bucket refreshes only when the new
    anchor cannot improve any point in the bucket.
    """
    _validate_points_and_m("_fps_exact_bucketed", points, int(M), int(seed), allow_batched=True)
    if bucket_size < 1:
        raise ValueError("_fps_exact_bucketed: bucket_size must be >= 1")

    points = _as_contiguous(points)
    was_single = points.dim() == 2
    points_batched = points.unsqueeze(0) if was_single else points
    seed_indices = _resolve_seed_batched(points_batched, int(seed)).contiguous()
    with torch.no_grad():
        sorted_indices = _C.morton_sort_points_batched(points_batched.detach()).contiguous()

    fps_idx, nearest_anchor, nearest_dist = _C.fps_exact_bucketed_lean(
        points_batched,
        seed_indices,
        sorted_indices,
        int(M),
        int(bucket_size),
        bool(enable_pruning),
        bool(use_graph),
    )
    anchor_radius, anchor_counts, fps_leaf_pos = _C.fps_metadata(
        points_batched,
        fps_idx,
        nearest_anchor,
        nearest_dist,
        sorted_indices,
    )
    result = _make_batched_result(
        points=points_batched,
        fps_idx=fps_idx,
        nearest_anchor=nearest_anchor,
        nearest_anchor_dist_sq=nearest_dist,
        anchor_radius=anchor_radius,
        anchor_counts=anchor_counts,
        fps_leaf_pos=fps_leaf_pos,
    )

    if was_single:
        result = FPSResult(
            indices=result.indices[0],
            points=result.points[0],
            nearest_anchor=result.nearest_anchor[0],
            nearest_anchor_dist_sq=result.nearest_anchor_dist_sq[0],
            anchor_radius=result.anchor_radius[0],
            anchor_counts=result.anchor_counts[0],
            coarse_order=result.coarse_order[0],
            selection_order_indices=result.selection_order_indices[0],
        )

    return result


def _fps_approx_bucketed(
    points: torch.Tensor,
    M: int,
    seed: int = 0,
    *,
    bucket_size: int = 256,
    refresh_interval: int = 8,
    candidates_per_round: int = 32,
    anchors_per_round: int = 4,
    alpha: float = 0.25,
    use_graph: bool = True,
) -> FPSResult:
    """Maintained approximate bucket-queue FPS CUDA route.

    This helper selects approximate anchors from compact Morton buckets,
    incrementally maintains exact nearest-anchor state for the selected anchors,
    and returns normal ``FPSResult`` metadata.
    """
    _validate_points_and_m("_fps_approx_bucketed", points, int(M), int(seed), allow_batched=True)
    if bucket_size < 1:
        raise ValueError("_fps_approx_bucketed: bucket_size must be >= 1")
    if refresh_interval < 1:
        raise ValueError("_fps_approx_bucketed: refresh_interval must be >= 1")
    if anchors_per_round < 1 or anchors_per_round > 8:
        raise ValueError("_fps_approx_bucketed: anchors_per_round must be in [1, 8]")
    if candidates_per_round < anchors_per_round or candidates_per_round > 32:
        raise ValueError("_fps_approx_bucketed: candidates_per_round must be in [anchors_per_round, 32]")
    if alpha < 0.0:
        raise ValueError("_fps_approx_bucketed: alpha must be nonnegative")

    points = _as_contiguous(points)
    was_single = points.dim() == 2
    points_batched = points.unsqueeze(0) if was_single else points
    seed_indices = _resolve_seed_batched(points_batched, int(seed)).contiguous()
    with torch.no_grad():
        sorted_indices = _C.morton_sort_points_batched(points_batched.detach()).contiguous()

    fps_idx, nearest_anchor, nearest_dist = _C.fps_approx_bucketed_lean(
        points_batched,
        seed_indices,
        sorted_indices,
        int(M),
        int(bucket_size),
        int(refresh_interval),
        int(candidates_per_round),
        int(anchors_per_round),
        float(alpha),
        use_graph=use_graph,
    )
    anchor_radius, anchor_counts, fps_leaf_pos = _C.fps_metadata(
        points_batched,
        fps_idx,
        nearest_anchor,
        nearest_dist,
        sorted_indices,
    )
    result = _make_batched_result(
        points=points_batched,
        fps_idx=fps_idx,
        nearest_anchor=nearest_anchor,
        nearest_anchor_dist_sq=nearest_dist,
        anchor_radius=anchor_radius,
        anchor_counts=anchor_counts,
        fps_leaf_pos=fps_leaf_pos,
    )

    if was_single:
        result = FPSResult(
            indices=result.indices[0],
            points=result.points[0],
            nearest_anchor=result.nearest_anchor[0],
            nearest_anchor_dist_sq=result.nearest_anchor_dist_sq[0],
            anchor_radius=result.anchor_radius[0],
            anchor_counts=result.anchor_counts[0],
            coarse_order=result.coarse_order[0],
            selection_order_indices=result.selection_order_indices[0],
        )

    return result


def fps(
    points: torch.Tensor,
    target_tokens: int,
    *,
    seed: int = 0,
    r: int = 4,
    c: int = 2,
    alpha: float = 0.25,
    mode: str = "exact_bucketed",
    bucket_size: int = 256,
    use_graph: bool = True,
) -> FPSResult:
    """Build FPS geometry for CUDA point clouds.

    ``mode`` selects the sampler implementation:

    - ``"exact_bucketed"``: default exact path using graph-captured bucket maxima.
    - ``"approx_bucketed"``: approximate bucket-queue path using ``r``, ``c``, and
      ``alpha``.
    """
    requested_mode = str(mode)
    if requested_mode not in ("exact_bucketed", "approx_bucketed"):
        raise ValueError("fps: mode must be one of 'exact_bucketed' or 'approx_bucketed'")
    _validate_points_and_m("fps", points, int(target_tokens), int(seed), allow_batched=True)
    points = _as_contiguous(points)

    if requested_mode == "exact_bucketed":
        return _fps_exact_bucketed(
            points,
            int(target_tokens),
            seed=int(seed),
            bucket_size=int(bucket_size),
            use_graph=bool(use_graph),
            enable_pruning=True,
        )
    if requested_mode == "approx_bucketed":
        if int(r) < 1 or int(r) > 8:
            raise ValueError("fps: r must be in [1, 8] for mode='approx_bucketed'")
        if int(c) < 1:
            raise ValueError("fps: c must be >= 1 for mode='approx_bucketed'")
        if int(r) * int(c) > 32:
            raise ValueError("fps: r * c must be <= 32 for mode='approx_bucketed'")
        return _fps_approx_bucketed(
            points,
            int(target_tokens),
            seed=int(seed),
            bucket_size=int(bucket_size),
            refresh_interval=8,
            candidates_per_round=max(int(r), int(r) * int(c)),
            anchors_per_round=int(r),
            alpha=float(alpha),
            use_graph=bool(use_graph),
        )
    raise AssertionError("unreachable FPS mode")
