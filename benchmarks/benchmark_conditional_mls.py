"""Baseline-versus-routed conditional MLS benchmark.

Defaults model the production workload B=16, N_field=M=16000,
N_boundary=1600, H=40. Timing-sensitive assertions intentionally do not live
in CI; this script records forward, forward/backward, and peak allocation.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torchbvh


def _time_cuda(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        value = fn()
        del value
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        value = fn()
        del value
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - start) / iterations


def _peak_mib(fn) -> float:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    value = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    del value
    return peak / (1024**2)


def _make_inputs(args, dim: int):
    torch.manual_seed(12000 + dim)
    B, M, H, C = args.batch, args.queries, args.heads, args.channels
    boundary_points = torch.rand((B, args.boundary_points, dim), device="cuda")
    field_points = torch.rand((B, args.field_points, dim), device="cuda")
    boundary_queries = torch.rand((B, M, H, dim), device="cuda", requires_grad=True)
    field_queries = torch.rand((B, M, H, dim), device="cuda", requires_grad=True)
    boundary_features = torch.rand(
        (B, args.boundary_points, H, C), device="cuda", requires_grad=True)
    field_features = torch.rand(
        (B, args.field_points, H, C), device="cuda", requires_grad=True)
    mask = torch.rand((B, M, H), device="cuda") < args.hit_rate
    return mask, boundary_points, boundary_queries, boundary_features, field_points, field_queries, field_features


def _conditional(inputs, k):
    mask, bp, bq, bf, fp, fq, ff = inputs
    return torchbvh.conditional_mls_interpolate(
        mask,
        true_points=bp,
        true_queries=bq,
        true_features=bf,
        false_points=fp,
        false_queries=fq,
        false_features=ff,
        k=k,
    )


def _baseline(inputs, k):
    mask, bp, bq, bf, fp, fq, ff = inputs
    true_values = torchbvh.bvh_mls_interpolate_batched_heads(bp, bq, bf, k=k)
    false_values = torchbvh.bvh_mls_interpolate_batched_heads(fp, fq, ff, k=k)
    return torch.where(mask[..., None], true_values, false_values)


def _with_backward(fn, inputs, k):
    for tensor in (inputs[2], inputs[3], inputs[5], inputs[6]):
        tensor.grad = None
    result = fn(inputs, k)
    result.sum().backward()
    return result


def _clear_grads(inputs):
    for tensor in (inputs[2], inputs[3], inputs[5], inputs[6]):
        tensor.grad = None


def _run_case(args, dim: int):
    inputs = _make_inputs(args, dim)
    with torch.no_grad():
        baseline_forward_ms = _time_cuda(
            lambda: _baseline(inputs, args.k), args.warmup, args.iterations)
        conditional_forward_ms = _time_cuda(
            lambda: _conditional(inputs, args.k), args.warmup, args.iterations)
        baseline_forward_peak_mib = _peak_mib(lambda: _baseline(inputs, args.k))
        conditional_forward_peak_mib = _peak_mib(lambda: _conditional(inputs, args.k))

    baseline_backward_ms = _time_cuda(
        lambda: _with_backward(_baseline, inputs, args.k),
        args.backward_warmup, args.backward_iterations)
    conditional_backward_ms = _time_cuda(
        lambda: _with_backward(_conditional, inputs, args.k),
        args.backward_warmup, args.backward_iterations)
    _clear_grads(inputs)
    baseline_backward_peak_mib = _peak_mib(
        lambda: _with_backward(_baseline, inputs, args.k))
    _clear_grads(inputs)
    conditional_backward_peak_mib = _peak_mib(
        lambda: _with_backward(_conditional, inputs, args.k))
    return {
        "spatial_dim": dim,
        "baseline_forward_ms": baseline_forward_ms,
        "conditional_forward_ms": conditional_forward_ms,
        "forward_speedup": baseline_forward_ms / conditional_forward_ms,
        "baseline_forward_backward_ms": baseline_backward_ms,
        "conditional_forward_backward_ms": conditional_backward_ms,
        "forward_backward_speedup": baseline_backward_ms / conditional_backward_ms,
        "baseline_forward_peak_mib": baseline_forward_peak_mib,
        "conditional_forward_peak_mib": conditional_forward_peak_mib,
        "forward_peak_reduction_mib": baseline_forward_peak_mib - conditional_forward_peak_mib,
        "baseline_forward_backward_peak_mib": baseline_backward_peak_mib,
        "conditional_forward_backward_peak_mib": conditional_backward_peak_mib,
        "forward_backward_peak_reduction_mib": baseline_backward_peak_mib - conditional_backward_peak_mib,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", choices=(2, 3), default=(2, 3))
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--field-points", type=int, default=16000)
    parser.add_argument("--boundary-points", type=int, default=1600)
    parser.add_argument("--queries", type=int, default=16000)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--hit-rate", type=float, default=0.2)
    parser.add_argument("--k", type=int, choices=(4, 8, 16), default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--backward-warmup", type=int, default=1)
    parser.add_argument("--backward-iterations", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_conditional_mls requires CUDA")
    if not 0.0 <= args.hit_rate <= 1.0:
        parser.error("--hit-rate must be in [0, 1]")

    results = [_run_case(args, dim) for dim in args.dims]
    print(json.dumps({
        "device": torch.cuda.get_device_name(),
        "batch": args.batch,
        "field_points": args.field_points,
        "boundary_points": args.boundary_points,
        "queries_per_head": args.queries,
        "heads": args.heads,
        "channels_per_head": args.channels,
        "hit_rate": args.hit_rate,
        "k": args.k,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
