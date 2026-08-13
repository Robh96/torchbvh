"""Small build/traversal/one-shot benchmark for the public ray API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torchbvh


def _time_cuda(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - start) / iterations


def _inputs(primitive_type: str, batch: int, primitives: int, rays: int):
    dim = 2 if primitive_type == "segment" else 3
    vertices = 2 if primitive_type == "segment" else 3
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primitive-type", choices=("segment", "triangle"), default="segment")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--primitives", type=int, default=4096)
    parser.add_argument("--rays", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--dense-max-pairs", type=int, default=10_000_000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_raytrace requires CUDA")
    geometry, origins, directions = _inputs(
        args.primitive_type, args.batch, args.primitives, args.rays
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

    print(json.dumps({
        "device": torch.cuda.get_device_name(),
        "primitive_type": args.primitive_type,
        "batch": args.batch,
        "primitive_count": args.primitives,
        "ray_count": args.rays,
        "build_ms": build_ms,
        "reused_traversal_ms": traversal_ms,
        "one_shot_ms": one_shot_ms,
        "dense_ms": dense_ms,
        "dense_skip": dense_skip,
    }, indent=2))


if __name__ == "__main__":
    main()
