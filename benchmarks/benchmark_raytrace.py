"""Small build/traversal/one-shot benchmark for the public ray API."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torchbvh
from torchbvh._ray import _selected_hit_t


def _time_cuda(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - start) / iterations


def _inputs(primitive_type: str, batch: int, primitives: int, rays: int, scene: str):
    dim = 2 if primitive_type == "segment" else 3
    vertices = 2 if primitive_type == "segment" else 3
    if scene != "random":
        if primitive_type != "triangle":
            raise ValueError("mesh and miss scenes require triangles")
        cells_per_axis = math.ceil(math.sqrt(primitives / 2))
        grid = torch.linspace(-1, 1, cells_per_axis + 1, device="cuda")
        x, y = torch.meshgrid(grid, grid, indexing="ij")
        z = torch.zeros_like(x)
        p00 = torch.stack((x[:-1, :-1], y[:-1, :-1], z[:-1, :-1]), dim=-1)
        p10 = torch.stack((x[1:, :-1], y[1:, :-1], z[1:, :-1]), dim=-1)
        p01 = torch.stack((x[:-1, 1:], y[:-1, 1:], z[:-1, 1:]), dim=-1)
        p11 = torch.stack((x[1:, 1:], y[1:, 1:], z[1:, 1:]), dim=-1)
        triangles = torch.stack((torch.stack((p00, p10, p01), dim=-2),
                                 torch.stack((p11, p01, p10), dim=-2)), dim=-3)
        geometry = triangles.reshape(-1, 3, 3)[:primitives].repeat(batch, 1, 1, 1)
        origins = torch.rand((batch, rays, 3), device="cuda") * 2 - 1
        origins[..., 2] = -0.5 if scene == "mesh" else 2.0
        directions = torch.zeros_like(origins)
        directions[..., 2] = 1.0
        return geometry, origins, directions
    geometry = torch.rand((batch, primitives, vertices, dim), device="cuda") * 2 - 1
    origins = torch.rand((batch, rays, dim), device="cuda") * 2 - 1
    directions = torch.rand((batch, rays, dim), device="cuda") * 2 - 1
    return geometry, origins, directions


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _dense_trace(primitive_type: str, geometry, origins, directions):
    if primitive_type == "segment":
        edge = geometry[:, None, :, 1] - geometry[:, None, :, 0]
        offset = geometry[:, None, :, 0] - origins[:, :, None]
        rays = directions[:, :, None]
        denominator = _cross2(rays, edge)
        valid_denominator = denominator.abs() > 8 * torch.finfo(torch.float32).eps
        safe_denominator = torch.where(valid_denominator, denominator, torch.ones_like(denominator))
        t = _cross2(offset, edge) / safe_denominator
        u = _cross2(offset, rays) / safe_denominator
        valid = valid_denominator & (t >= 1e-7) & (t <= 1.0) & (u >= 0) & (u <= 1)
    else:
        vertex0 = geometry[:, None, :, 0]
        edge1 = geometry[:, None, :, 1] - vertex0
        edge2 = geometry[:, None, :, 2] - vertex0
        rays = directions[:, :, None]
        pvec = torch.linalg.cross(rays, edge2, dim=-1)
        determinant = (edge1 * pvec).sum(-1)
        valid_determinant = determinant.abs() > 8 * torch.finfo(torch.float32).eps
        inverse = torch.where(valid_determinant, determinant.reciprocal(), torch.zeros_like(determinant))
        tvec = origins[:, :, None] - vertex0
        u = (tvec * pvec).sum(-1) * inverse
        qvec = torch.linalg.cross(tvec, edge1, dim=-1)
        v = (rays * qvec).sum(-1) * inverse
        t = (edge2 * qvec).sum(-1) * inverse
        valid = valid_determinant & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 1e-7) & (t <= 1.0)
    return torch.where(valid, t, torch.full_like(t, float("inf"))).min(dim=-1)


def _general_triangle_trace(bvh, geometry, origins, directions):
    """Retained pre-optimization triangle path for same-process comparisons."""
    data = bvh._data
    indices, cuda_t = torchbvh._C.raytrace_batched(
        data["node_aabbs"], data["sorted_indices"], data["left_child_mem"],
        data["right_child_mem"], data["mem_to_leaf"], geometry.detach().contiguous(),
        origins.detach(), directions.detach(), data["num_real_nodes"], 1e-7, 1.0,
    )
    hit_t, points, mask = _selected_hit_t(
        geometry, origins, directions, indices.detach(), cuda_t.detach(), "triangle"
    )
    return indices, hit_t, points, mask


def _compare_triangle_paths(geometry, origins, directions, warmup, iterations):
    with torchbvh.RayBVH(geometry, primitive_type="triangle") as bvh:
        old = _general_triangle_trace(bvh, geometry, origins, directions)
        new = bvh.trace(origins, directions, t_max=1.0)
        torch.testing.assert_close(new.primitive_indices, old[0])
        torch.testing.assert_close(new.t[new.mask], old[1][old[3]], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(new.points[new.mask], old[2][old[3]], rtol=1e-5, atol=1e-5)
        assert torch.equal(new.mask, old[3])

        old_times = []
        new_times = []
        for repeat in range(5):
            measurements = (
                (("old", lambda: _general_triangle_trace(bvh, geometry, origins, directions)),
                 ("new", lambda: bvh.trace(origins, directions, t_max=1.0)))
                if repeat % 2 == 0 else
                (("new", lambda: bvh.trace(origins, directions, t_max=1.0)),
                 ("old", lambda: _general_triangle_trace(bvh, geometry, origins, directions)))
            )
            for label, fn in measurements:
                (old_times if label == "old" else new_times).append(
                    _time_cuda(fn, warmup, iterations)
                )

    grad_geometry = geometry.detach().requires_grad_()
    grad_origins = origins.detach().requires_grad_()
    grad_directions = directions.detach().requires_grad_()
    with torchbvh.RayBVH(grad_geometry, primitive_type="triangle") as bvh:
        def run_backward(use_general):
            grad_geometry.grad = None
            grad_origins.grad = None
            grad_directions.grad = None
            if use_general:
                _, hit_t, points, mask = _general_triangle_trace(
                    bvh, grad_geometry, grad_origins, grad_directions
                )
            else:
                result = bvh.trace(grad_origins, grad_directions, t_max=1.0)
                hit_t, points, mask = result.t, result.points, result.mask
            loss = torch.where(mask, hit_t, 0).sum() + torch.where(mask[..., None], points, 0).sum()
            loss.backward()

        old_backward_times = []
        new_backward_times = []
        for repeat in range(5):
            for label, use_general in ((("old", True), ("new", False)) if repeat % 2 == 0
                                       else (("new", False), ("old", True))):
                (old_backward_times if use_general else new_backward_times).append(
                    _time_cuda(lambda: run_backward(use_general), warmup, iterations)
                )

    old_ms = statistics.median(old_times)
    new_ms = statistics.median(new_times)
    old_backward_ms = statistics.median(old_backward_times)
    new_backward_ms = statistics.median(new_backward_times)
    return {
        "general_reused_ms": old_ms,
        "cached_reused_ms": new_ms,
        "forward_speedup": old_ms / new_ms,
        "general_forward_backward_ms": old_backward_ms,
        "cached_forward_backward_ms": new_backward_ms,
        "forward_backward_speedup": old_backward_ms / new_backward_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primitive-type", choices=("segment", "triangle"), default="segment")
    parser.add_argument("--scene", choices=("random", "mesh", "miss"), default="random")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--primitives", type=int, default=4096)
    parser.add_argument("--rays", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--dense-max-pairs", type=int, default=10_000_000)
    parser.add_argument("--compare-general", action="store_true",
                        help="Compare cached triangles with the retained general 3-D path")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_raytrace requires CUDA")
    if args.primitive_type == "segment" and args.scene != "random":
        parser.error("--scene mesh/miss requires --primitive-type triangle")
    if args.compare_general and args.primitive_type != "triangle":
        parser.error("--compare-general requires --primitive-type triangle")
    geometry, origins, directions = _inputs(
        args.primitive_type, args.batch, args.primitives, args.rays, args.scene
    )

    build_ms = _time_cuda(
        lambda: torchbvh.RayBVH(geometry, primitive_type=args.primitive_type).destroy(),
        args.warmup,
        args.iterations,
    )
    with torchbvh.RayBVH(geometry, primitive_type=args.primitive_type) as bvh:
        traversal_ms = _time_cuda(
            lambda: bvh.trace(origins, directions, t_max=1.0),
            args.warmup,
            args.iterations,
        )
    one_shot_ms = _time_cuda(
        lambda: torchbvh.raytrace(
            geometry, origins, directions,
            primitive_type=args.primitive_type, t_max=1.0,
        ),
        args.warmup,
        args.iterations,
    )

    pair_count = args.batch * args.primitives * args.rays
    dense_ms = None
    dense_skip = None
    if pair_count <= args.dense_max_pairs:
        dense_ms = _time_cuda(
            lambda: _dense_trace(args.primitive_type, geometry, origins, directions),
            args.warmup,
            args.iterations,
        )
    else:
        dense_skip = f"{pair_count} pairs exceed --dense-max-pairs={args.dense_max_pairs}"

    report = {
        "device": torch.cuda.get_device_name(),
        "primitive_type": args.primitive_type,
        "scene": args.scene,
        "batch": args.batch,
        "primitive_count": args.primitives,
        "ray_count": args.rays,
        "build_ms": build_ms,
        "reused_traversal_ms": traversal_ms,
        "one_shot_ms": one_shot_ms,
        "dense_ms": dense_ms,
        "dense_skip": dense_skip,
    }
    if args.compare_general:
        report.update(_compare_triangle_paths(
            geometry, origins, directions, args.warmup, args.iterations
        ))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
