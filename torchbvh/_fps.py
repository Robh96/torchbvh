from __future__ import annotations

from dataclasses import dataclass

import torch

from . import _C
from ._query import build_bvh, build_bvh_batched


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
    if not points.is_contiguous():
        raise ValueError(f"{prefix}: points must be contiguous")
    if M < 1 or M > points.size(n_axis):
        raise ValueError(f"{prefix}: target token count must be in [1, N]")
    if seed < -1 or seed >= points.size(n_axis):
        raise ValueError(f"{prefix}: seed must be -1 or an original point index in [0, N)")


def _resolve_seed_batched(points: torch.Tensor, seed: int) -> torch.Tensor:
    if seed != -1:
        return torch.full((points.size(0),), int(seed), device=points.device, dtype=torch.int64)
    center = 0.5 * (points.min(dim=1).values + points.max(dim=1).values)
    return (points - center[:, None, :]).square().sum(dim=2).argmin(dim=1).to(torch.int64)


def _make_result(
    points: torch.Tensor,
    fps_idx: torch.Tensor,
    nearest_anchor: torch.Tensor,
    nearest_anchor_dist_sq: torch.Tensor,
    sorted_indices: torch.Tensor | None = None,
    anchor_radius: torch.Tensor | None = None,
    anchor_counts: torch.Tensor | None = None,
    fps_leaf_pos: torch.Tensor | None = None,
) -> FPSResult:
    if points.dim() == 2:
        M = int(fps_idx.numel())
        if anchor_radius is None or anchor_counts is None:
            anchor_radius = torch.zeros(M, device=points.device, dtype=points.dtype)
            anchor_counts = torch.bincount(nearest_anchor.to(torch.int64), minlength=M).to(torch.int32)
            for anchor in range(M):
                mask = nearest_anchor == anchor
                if bool(mask.any()):
                    anchor_radius[anchor] = nearest_anchor_dist_sq[mask].max()

        if fps_leaf_pos is None:
            if sorted_indices is None:
                with torch.no_grad():
                    bvh = build_bvh(points)
                    sorted_indices = bvh["sorted_indices"]

            reverse_leaf_pos = torch.empty(points.size(0), device=points.device, dtype=torch.int64)
            reverse_leaf_pos[sorted_indices] = torch.arange(points.size(0), device=points.device)
            fps_leaf_pos = reverse_leaf_pos[fps_idx]
        coarse_order = torch.argsort(fps_leaf_pos, stable=True)

        return FPSResult(
            indices=fps_idx,
            points=points[fps_idx],
            nearest_anchor=nearest_anchor.to(torch.int32),
            nearest_anchor_dist_sq=nearest_anchor_dist_sq,
            anchor_radius=anchor_radius,
            anchor_counts=anchor_counts,
            coarse_order=coarse_order,
            selection_order_indices=fps_idx,
        )

    B, N, D = points.shape
    M = int(fps_idx.size(1))
    if anchor_radius is None or anchor_counts is None:
        anchor_radius = torch.zeros((B, M), device=points.device, dtype=points.dtype)
        anchor_counts = torch.empty((B, M), device=points.device, dtype=torch.int32)
        for b in range(B):
            anchor_counts[b] = torch.bincount(nearest_anchor[b].to(torch.int64), minlength=M).to(torch.int32)
            for anchor in range(M):
                mask = nearest_anchor[b] == anchor
                if bool(mask.any()):
                    anchor_radius[b, anchor] = nearest_anchor_dist_sq[b, mask].max()

    if fps_leaf_pos is None:
        if sorted_indices is None:
            with torch.no_grad():
                bvh = build_bvh_batched(points)
                sorted_indices = bvh["sorted_indices"]

        reverse_leaf_pos = torch.empty((B, N), device=points.device, dtype=torch.int64)
        leaf_pos = torch.arange(N, device=points.device, dtype=torch.int64).expand(B, N)
        reverse_leaf_pos.scatter_(1, sorted_indices, leaf_pos)
        fps_leaf_pos = torch.gather(reverse_leaf_pos, 1, fps_idx)
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


def _fps_exact_full_scan(points: torch.Tensor, M: int, seed: int = 0) -> FPSResult:
    """Exact persistent full-scan FPS in original point order."""
    _validate_points_and_m(
        "_fps_exact_full_scan",
        points,
        int(M),
        int(seed),
        allow_batched=True,
    )
    was_single = points.dim() == 2
    points_batched = points.unsqueeze(0) if was_single else points
    seed_indices = _resolve_seed_batched(points_batched, int(seed)).contiguous()
    fps_idx, nearest_anchor, nearest_dist = _C.fps_exact_full_scan(
        points_batched,
        seed_indices,
        int(M),
    )
    with torch.no_grad():
        bvh = build_bvh_batched(points_batched)
        sorted_indices = bvh["sorted_indices"]
    anchor_radius, anchor_counts, fps_leaf_pos = _C.fps_metadata(
        points_batched,
        fps_idx,
        nearest_anchor,
        nearest_dist,
        sorted_indices,
    )
    result = _make_result(
        points_batched,
        fps_idx,
        nearest_anchor,
        nearest_dist,
        sorted_indices,
        anchor_radius,
        anchor_counts,
        fps_leaf_pos,
    )
    if not was_single:
        return result
    return FPSResult(
        indices=result.indices[0],
        points=result.points[0],
        nearest_anchor=result.nearest_anchor[0],
        nearest_anchor_dist_sq=result.nearest_anchor_dist_sq[0],
        anchor_radius=result.anchor_radius[0],
        anchor_counts=result.anchor_counts[0],
        coarse_order=result.coarse_order[0],
        selection_order_indices=result.selection_order_indices[0],
    )


def _fps_exact_bucketed(
    points: torch.Tensor,
    M: int,
    seed: int = 0,
    *,
    bucket_size: int = 256,
    use_graph: bool = True,
    enable_pruning: bool = True,
    return_diagnostics: bool = False,
) -> FPSResult | tuple[FPSResult, dict[str, int | float | bool]]:
    """Maintained exact bucketed FPS CUDA route.

    Anchor selection remains exact: one anchor is selected per round from exact
    per-bucket maxima. AABB pruning can skip bucket refreshes only when the new
    anchor cannot improve any point in the bucket.
    """
    _validate_points_and_m("_fps_exact_bucketed", points, int(M), int(seed), allow_batched=True)
    if bucket_size < 1:
        raise ValueError("_fps_exact_bucketed: bucket_size must be >= 1")

    was_single = points.dim() == 2
    points_batched = points.unsqueeze(0) if was_single else points
    seed_indices = _resolve_seed_batched(points_batched, int(seed)).contiguous()
    with torch.no_grad():
        bvh = build_bvh_batched(points_batched)

    out = _C.fps_exact_bucketed(
        points_batched,
        seed_indices,
        bvh["sorted_indices"],
        bvh["left_child_mem"],
        bvh["right_child_mem"],
        bvh["mem_to_leaf"],
        bvh["node_aabbs"],
        int(bvh["num_real_nodes"]),
        int(bvh["leaf_level"]),
        int(M),
        int(bucket_size),
        use_graph=bool(use_graph),
        enable_pruning=bool(enable_pruning),
    )
    fps_idx, nearest_anchor, nearest_dist, refreshed_counts, skipped_counts, bucket_info = out
    anchor_radius, anchor_counts, fps_leaf_pos = _C.fps_metadata(
        points_batched,
        fps_idx,
        nearest_anchor,
        nearest_dist,
        bvh["sorted_indices"],
    )
    result = _make_result(
        points=points_batched,
        fps_idx=fps_idx,
        nearest_anchor=nearest_anchor,
        nearest_anchor_dist_sq=nearest_dist,
        anchor_radius=anchor_radius,
        anchor_counts=anchor_counts,
        fps_leaf_pos=fps_leaf_pos,
    )

    bucket_info_cpu = bucket_info.detach().cpu()
    total_refreshes = int(refreshed_counts.sum().item())
    total_skips = int(skipped_counts.sum().item())
    round_slots = max((int(M) - 1) * points_batched.size(0), 1)
    init_refreshes = points_batched.size(0) * int(bucket_info_cpu[1].item())
    per_round_refreshes = max(total_refreshes - init_refreshes, 0)
    mean_refreshes = per_round_refreshes / round_slots if M > 1 else 0.0
    mean_skips = total_skips / round_slots if M > 1 else 0.0
    diagnostics: dict[str, int | float | bool] = {
        "route": int(bucket_info_cpu[9].item()),
        "bucket_level": int(bucket_info_cpu[0].item()),
        "bucket_count": int(bucket_info_cpu[1].item()),
        "bucket_size_min": int(bucket_info_cpu[2].item()),
        "bucket_size_max": int(bucket_info_cpu[3].item()),
        "bucket_size_requested": int(bucket_info_cpu[4].item()),
        "enable_pruning": bool(bucket_info_cpu[5].item()),
        "use_graph_requested": bool(bucket_info_cpu[6].item()),
        "graph_captured": bool(bucket_info_cpu[7].item()),
        "total_refreshes": total_refreshes,
        "total_skips": total_skips,
        "mean_refreshed_buckets_per_round": float(mean_refreshes),
        "mean_skipped_buckets_per_round": float(mean_skips),
        "mean_refreshed_bucket_ratio": float(
            mean_refreshes / max(int(bucket_info_cpu[1].item()), 1)
        ) if M > 1 else 0.0,
        "round_slots": round_slots,
    }

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

    if return_diagnostics:
        return result, diagnostics
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
    return_diagnostics: bool = False,
) -> FPSResult | tuple[FPSResult, dict[str, int | float]]:
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

    was_single = points.dim() == 2
    points_batched = points.unsqueeze(0) if was_single else points
    seed_indices = _resolve_seed_batched(points_batched, int(seed)).contiguous()
    with torch.no_grad():
        bvh = build_bvh_batched(points_batched)

    out = _C.fps_approx_bucketed(
        points_batched,
        seed_indices,
        bvh["sorted_indices"],
        bvh["left_child_mem"],
        bvh["right_child_mem"],
        bvh["mem_to_leaf"],
        bvh["node_aabbs"],
        int(bvh["num_real_nodes"]),
        int(bvh["leaf_level"]),
        int(M),
        int(bucket_size),
        int(refresh_interval),
        int(candidates_per_round),
        int(anchors_per_round),
        float(alpha),
        use_graph=use_graph,
    )
    (
        fps_idx,
        nearest_anchor,
        nearest_dist,
        mode_flags,
        visited_leaf_counts,
        committed_counts,
        candidate_counts,
        rejected_counts,
        refresh_flags,
        bucket_info,
    ) = out
    anchor_radius, anchor_counts, fps_leaf_pos = _C.fps_metadata(
        points_batched,
        fps_idx,
        nearest_anchor,
        nearest_dist,
        bvh["sorted_indices"],
    )
    result = _make_result(
        points=points_batched,
        fps_idx=fps_idx,
        nearest_anchor=nearest_anchor,
        nearest_anchor_dist_sq=nearest_dist,
        anchor_radius=anchor_radius,
        anchor_counts=anchor_counts,
        fps_leaf_pos=fps_leaf_pos,
    )

    bucket_info_cpu = bucket_info.detach().cpu()
    active_commits = committed_counts[committed_counts > 0].float()
    active_candidates = candidate_counts[candidate_counts > 0].float()
    diagnostics: dict[str, int | float] = {
        "route": 2,
        "bucket_level": int(bucket_info_cpu[0].item()),
        "bucket_count": int(bucket_info_cpu[1].item()),
        "bucket_size_min": int(bucket_info_cpu[2].item()),
        "bucket_size_max": int(bucket_info_cpu[3].item()),
        "bucket_size_requested": int(bucket_info_cpu[4].item()),
        "refresh_interval": int(bucket_info_cpu[5].item()),
        "candidates_per_round": int(bucket_info_cpu[6].item()),
        "anchors_per_round": int(bucket_info_cpu[7].item()),
        "approx_iterations": int(bucket_info_cpu[8].item()),
        "bucket_refresh_work": int(bucket_info_cpu[9].item()),
        "top_risk_buckets": int(bucket_info_cpu[11].item()),
        "dirty_refresh_interval": int(bucket_info_cpu[12].item()),
        "anchors_committed": int(committed_counts.sum().item()),
        "candidates_generated": int(candidate_counts.sum().item()),
        "duplicate_rejects": int(rejected_counts.sum().item()),
        "global_refresh_count": int(refresh_flags.sum().item()),
        "native_bucket_rounds": int((mode_flags[:, 1:] == 3).sum().item()),
        "mean_visited_leaf_ratio": float((visited_leaf_counts[:, 1:].float() / points_batched.size(1)).mean().item()),
        "committed_mean": float(active_commits.mean().item()) if active_commits.numel() else 0.0,
        "candidate_mean": float(active_candidates.mean().item()) if active_candidates.numel() else 0.0,
    }

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

    if return_diagnostics:
        return result, diagnostics
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
    - ``"exact_full_scan"``: exact full-scan fallback with lower persistent memory.
    - ``"approx_bucketed"``: approximate bucket-queue path using ``r``, ``c``, and
      ``alpha``.
    """
    _validate_points_and_m("fps", points, int(target_tokens), int(seed), allow_batched=True)
    requested_mode = str(mode)

    if requested_mode == "exact_bucketed":
        return _fps_exact_bucketed(
            points,
            int(target_tokens),
            seed=int(seed),
            bucket_size=int(bucket_size),
            use_graph=bool(use_graph),
            enable_pruning=True,
            return_diagnostics=False,
        )
    if requested_mode == "exact_full_scan":
        return _fps_exact_full_scan(points, int(target_tokens), seed=int(seed))
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
            return_diagnostics=False,
        )
    raise ValueError(
        "fps: mode must be one of "
        "'exact_bucketed', 'exact_full_scan', or 'approx_bucketed'"
    )
