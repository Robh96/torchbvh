"""Small, maintained timing sweep over the public production routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _time_cuda(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - start) / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=1, help="MLS heads and k-NN query multiplier")
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--dim", type=int, choices=(2, 3), default=3)
    parser.add_argument("--k", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--primitives", type=int, default=2048)
    parser.add_argument("--rays", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--use-graph", action="store_true", help="enable FPS CUDA graph capture")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("a CUDA-enabled PyTorch build and NVIDIA GPU are required")
    if min(args.batch, args.heads, args.channels) < 1:
        parser.error("batch, heads, and channels must be positive")
    if args.points < args.k or not 1 <= args.tokens <= args.points:
        parser.error("require --points >= --k and 1 <= --tokens <= --points")
    if min(args.queries, args.primitives, args.rays, args.iterations) < 1 or args.warmup < 0:
        parser.error("queries, primitives, rays, and iterations must be positive; warmup nonnegative")

    import torchbvh

    torch.manual_seed(0)
    point_bank = torch.rand((args.batch, args.points, args.dim), device="cuda")
    query_bank = torch.rand((args.batch, args.queries * args.heads, args.dim), device="cuda")
    feature_bank = torch.rand((args.batch, args.points, args.channels), device="cuda")
    points = point_bank[0] if args.batch == 1 else point_bank
    queries = query_bank[0] if args.batch == 1 else query_bank
    features = feature_bank[0] if args.batch == 1 else feature_bank
    timings = {}

    def build_once():
        handle = torchbvh.build_bvh(points)
        handle.destroy()

    timings["build_ms"] = _time_cuda(build_once, args.warmup, args.iterations)
    handle = torchbvh.build_bvh(points)
    try:
        timings[f"knn_k{args.k}_ms"] = _time_cuda(
            lambda: torchbvh.query_knn(handle, queries, args.k), args.warmup, args.iterations
        )
    finally:
        handle.destroy()
    if args.heads == 1:
        mls = lambda: torchbvh.mls_interpolate(points, queries, features, k=args.k)
    else:
        mls_queries = torch.rand((args.batch, args.queries, args.heads, args.dim), device="cuda")
        mls_features = torch.rand(
            (args.batch, args.points, args.heads, args.channels), device="cuda"
        )
        mls = lambda: torchbvh.bvh_mls_interpolate_batched_heads(
            point_bank, mls_queries, mls_features, k=args.k
        )
    timings[f"mls_k{args.k}_ms"] = _time_cuda(mls, args.warmup, args.iterations)
    for mode in ("exact_bucketed", "approx_bucketed"):
        timings[f"fps_{mode}_ms"] = _time_cuda(
            lambda mode=mode: torchbvh.fps(
                points, args.tokens, mode=mode, use_graph=args.use_graph
            ),
            args.warmup,
            args.iterations,
        )

    for primitive_type, dim, vertices in (("segment", 2, 2), ("triangle", 3, 3)):
        geometry_bank = torch.rand((args.batch, args.primitives, vertices, dim), device="cuda")
        origin_bank = torch.rand((args.batch, args.rays, dim), device="cuda")
        direction_bank = torch.rand((args.batch, args.rays, dim), device="cuda") - 0.5
        geometry = geometry_bank[0] if args.batch == 1 else geometry_bank
        origins = origin_bank[0] if args.batch == 1 else origin_bank
        directions = direction_bank[0] if args.batch == 1 else direction_bank
        timings[f"ray_{primitive_type}_ms"] = _time_cuda(
            lambda: torchbvh.raytrace(
                geometry, origins, directions, primitive_type=primitive_type
            ),
            args.warmup,
            args.iterations,
        )

    print(json.dumps({
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "timing_method": "synchronized wall-clock milliseconds per call",
        "args": vars(args),
        "timings": timings,
    }, indent=2))


if __name__ == "__main__":
    main()
