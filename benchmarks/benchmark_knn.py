"""Benchmark BVH k-NN against the current CPU cKDTree path."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from collections.abc import Mapping
from typing import Any


SUPPORTED_K = (4, 8, 16)
SUPPORTED_DIMS = (2, 3)
SUPPORTED_DISTRIBUTIONS = (
    "uniform",
    "duplicate-heavy",
    "degenerate-axis",
    "planar-3d",
    "collinear-3d",
    "clustered",
    "outlier-heavy",
)
BENCHMARK_SCRIPT_VERSION = "stage13-m6"
FLOAT32_BYTES = 4
INT64_BYTES = 8
BOOL_BYTES = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1, help="fixed batch size for native batched benchmark mode")
    parser.add_argument("--n", type=int, default=10_000, help="number of source points")
    parser.add_argument("--queries", type=int, default=10_000, help="number of query points")
    parser.add_argument("--dim", type=int, nargs="+", default=list(SUPPORTED_DIMS), choices=SUPPORTED_DIMS)
    parser.add_argument("--k", type=int, nargs="+", default=list(SUPPORTED_K), choices=SUPPORTED_K)
    parser.add_argument("--iters", type=int, default=5, help="timed iterations per case")
    parser.add_argument("--warmup", type=int, default=2, help="untimed warmup iterations per case")
    parser.add_argument("--features", type=int, default=8, help="feature channels for optional MLS timing")
    parser.add_argument("--distribution", choices=SUPPORTED_DISTRIBUTIONS, default="uniform")
    parser.add_argument("--ragged-n", type=int, nargs="+", default=None, help="per-sample source counts for ragged mode")
    parser.add_argument(
        "--ragged-queries",
        type=int,
        nargs="+",
        default=None,
        help="per-sample query counts for ragged mode",
    )
    parser.add_argument("--displaced-query", action="store_true", help="measure benchmark-local multihead displaced-query path")
    parser.add_argument("--displaced-query-heads", type=int, default=1, help="displaced-query heads H")
    parser.add_argument("--displaced-query-channels", type=int, default=None, help="displaced-query value channels Ch")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-mls", action="store_true", help="skip GPU MLS solve timing")
    parser.add_argument("--no-cpu-baseline", action="store_true", help="skip the CPU scipy cKDTree baseline")
    parser.add_argument(
        "--gpu-sort-queries",
        action="store_true",
        help="also report split timings for torchbvh Morton-sorted ordered-query traversal",
    )
    parser.add_argument("--case-name", type=str, default=None, help="optional Stage 8 workload case name for result metadata")
    parser.add_argument("--comparison-label", type=str, default=None, help="optional Stage 8 comparison label for result metadata")
    parser.add_argument(
        "--torch-cluster-baseline",
        action="store_true",
        help="also time torch_cluster.knn and dense-output conversion for comparable CUDA k-NN modes",
    )
    parser.add_argument(
        "--cupy-knn-baseline",
        action="store_true",
        help="also time cupy_knn.LBVHIndex for comparable 3D CUDA k-NN modes",
    )
    parser.add_argument("--cupy-knn-leaf-size", type=int, default=32, help="cupy_knn LBVH leaf size")
    parser.add_argument("--cupy-knn-compact", action="store_true", help="enable cupy_knn tree compaction")
    parser.add_argument(
        "--cupy-knn-shrink-to-fit",
        action="store_true",
        help="shrink compacted cupy_knn tree storage to fit",
    )
    parser.add_argument(
        "--cupy-knn-sort-queries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="sort cupy_knn queries by Morton code before query traversal",
    )
    parser.add_argument(
        "--cupy-knn-sort-mode",
        choices=("configured", "sorted", "unsorted", "both"),
        default="configured",
        help="which cupy_knn query-sort mode to time; configured uses --cupy-knn-sort-queries",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="write full benchmark results to JSON")
    args = parser.parse_args(argv)
    validate_args(args)
    args.dim = _unique_in_order(args.dim)
    args.k = _unique_in_order(args.k)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.n < max(args.k):
        raise SystemExit("--n must be at least the largest requested --k")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.queries < 1:
        raise SystemExit("--queries must be positive")
    if args.iters < 1:
        raise SystemExit("--iters must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.features < 1:
        raise SystemExit("--features must be positive")
    if args.ragged_n is not None or args.ragged_queries is not None:
        if args.ragged_n is None or args.ragged_queries is None:
            raise SystemExit("--ragged-n and --ragged-queries must be provided together")
        if len(args.ragged_n) != len(args.ragged_queries):
            raise SystemExit("--ragged-n and --ragged-queries must have the same length")
        if any(n < max(args.k) for n in args.ragged_n):
            raise SystemExit("each --ragged-n value must be at least the largest requested --k")
        if any(q < 1 for q in args.ragged_queries):
            raise SystemExit("each --ragged-queries value must be positive")
        if args.batch_size != 1:
            raise SystemExit("--batch-size is not used with ragged mode")
    if args.displaced_query:
        if args.ragged_n is not None:
            raise SystemExit("--displaced-query is only supported for fixed-size batched mode")
        if args.batch_size < 1:
            raise SystemExit("--displaced-query requires a positive --batch-size")
        if args.displaced_query_heads < 1:
            raise SystemExit("--displaced-query-heads must be positive")
        if args.displaced_query_channels is None:
            raise SystemExit("--displaced-query-channels is required with --displaced-query")
        if args.displaced_query_channels < 1:
            raise SystemExit("--displaced-query-channels must be positive")
    if args.ragged_n is not None and args.torch_cluster_baseline:
        raise SystemExit("--torch-cluster-baseline is not supported with ragged mode")
    if args.cupy_knn_baseline:
        if any(dim != 3 for dim in args.dim):
            raise SystemExit("--cupy-knn-baseline is only supported with --dim 3")
        if args.ragged_n is not None:
            raise SystemExit("--cupy-knn-baseline is not supported with ragged mode")
        if args.cupy_knn_leaf_size < 1:
            raise SystemExit("--cupy-knn-leaf-size must be positive")
        if args.cupy_knn_shrink_to_fit and not args.cupy_knn_compact:
            raise SystemExit("--cupy-knn-shrink-to-fit requires --cupy-knn-compact")


def _unique_in_order(values: list[int]) -> list[int]:
    seen = set()
    unique = []
    for value in values:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def summarize_timings(samples: dict[str, list[float]]) -> dict[str, dict[str, float | None]]:
    return {
        name: {
            "mean_ms": _mean(values),
            "stdev_ms": _stdev(values),
            "min_ms": min(values) if values else None,
            "max_ms": max(values) if values else None,
        }
        for name, values in samples.items()
    }


def _query_count_config(args: argparse.Namespace) -> dict[str, Any]:
    requested_queries_per_sample = args.queries
    if args.ragged_queries is not None:
        effective_queries_per_sample: int | list[int] = args.ragged_queries
        effective_total_queries = sum(args.ragged_queries)
        query_count_source = "ragged_queries"
        query_count_note = "--ragged-queries controls the measured packed query counts; --queries is retained as CLI metadata."
    elif args.displaced_query:
        effective_queries_per_sample = args.n * args.displaced_query_heads
        effective_total_queries = args.batch_size * effective_queries_per_sample
        query_count_source = "displaced_query_flattened_n_times_heads"
        query_count_note = "--displaced-query measures flattened N * H displaced queries per sample; --queries is retained as CLI metadata."
    else:
        effective_queries_per_sample = args.queries
        effective_total_queries = args.batch_size * args.queries
        query_count_source = "queries"
        query_count_note = "--queries controls the measured query count per sample."
    return {
        # Backward-compatible key retained for existing benchmark consumers.
        "requested_queries": args.queries,
        "requested_queries_per_sample": requested_queries_per_sample,
        "effective_queries_per_sample": effective_queries_per_sample,
        "effective_total_queries": effective_total_queries,
        "query_count_source": query_count_source,
        "query_count_note": query_count_note,
    }


def _effective_query_config(args: argparse.Namespace) -> dict[str, Any]:
    """Backward-compatible alias for tests and older benchmark callers."""

    return _query_count_config(args)


def _top_level_query_count_config(args: argparse.Namespace) -> dict[str, Any]:
    return _query_count_config(args)


def _default_case_name(mode: str) -> str:
    return {
        "single": "benchmark_single",
        "fixed_batched": "benchmark_fixed_batched",
        "ragged": "benchmark_ragged",
        "displaced_query": "benchmark_displaced",
    }.get(mode, f"benchmark_{mode}")


def _default_comparison_label(mode: str) -> str:
    return {
        "single": "cpu_ckdtree",
        "fixed_batched": "single_sample_loop_vs_native_batched",
        "ragged": "none",
        "displaced_query": "none",
    }.get(mode, "none")


def _precision_policy() -> dict[str, str]:
    return {
        "coordinate_dtype": "float32",
        "feature_dtype": "float32",
        "value_dtype": "float32",
        "autocast": "disabled",
        "morton_build_coordinates": "float32",
    }


def _geometry_lifecycle_policy(mode: str) -> dict[str, Any]:
    return {
        "bvh_rebuild": "per_warmup_and_timed_iteration",
        "handle_destroy": "explicit_after_iteration",
        "resolution_geometry": "not_applicable",
        "geometry_changes_across_iterations": False,
    }


def _stage8_case_metadata(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    return {
        "case_name": args.case_name or _default_case_name(mode),
        "result_source": "benchmark",
        "comparison_label": args.comparison_label or _default_comparison_label(mode),
        "comparison_scope": "timing_only",
        "precision_policy": _precision_policy(),
        "geometry_lifecycle": _geometry_lifecycle_policy(mode),
    }


def _resolution_geometry_policy(mode: str) -> dict[str, Any]:
    return {
        "resolution_geometry_constructed": False,
        "reuse_scope": "not_applicable",
    }


def _tensor_nbytes(tensor) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def _add_tensor_bytes(footprint: dict[str, int], family: str, tensor) -> None:
    footprint[family] += _tensor_nbytes(tensor)


def _footprint_total(footprint: dict[str, int]) -> dict[str, int]:
    footprint["total_bytes"] = sum(value for key, value in footprint.items() if key != "total_bytes")
    return footprint


def _query_output_footprint(
    *,
    batch_size: int,
    queries_per_sample: int,
    k: int,
    dim: int,
    include_source_position_gather: bool = True,
) -> dict[str, int]:
    query_neighbors = batch_size * queries_per_sample * k
    footprint = {
        "indices_bytes": query_neighbors * INT64_BYTES,
        "squared_distances_bytes": query_neighbors * FLOAT32_BYTES,
        "source_position_gather_bytes": (
            query_neighbors * dim * FLOAT32_BYTES if include_source_position_gather else 0
        ),
    }
    return _footprint_total(footprint)


def _mls_tensor_footprint(
    *,
    batch_size: int,
    queries_per_sample: int,
    k: int,
    dim: int,
    channels: int,
) -> dict[str, int]:
    query_neighbors = batch_size * queries_per_sample * k
    queries = batch_size * queries_per_sample
    basis_dim = dim + 1
    footprint = {
        "neighbor_feature_gather_bytes": query_neighbors * channels * FLOAT32_BYTES,
        "delta_bytes": query_neighbors * dim * FLOAT32_BYTES,
        "weights_bytes": query_neighbors * FLOAT32_BYTES,
        "basis_bytes": query_neighbors * basis_dim * FLOAT32_BYTES,
        "normal_matrix_bytes": queries * basis_dim * basis_dim * FLOAT32_BYTES,
        "rhs_bytes": queries * basis_dim * channels * FLOAT32_BYTES,
        "coefficients_bytes": queries * basis_dim * channels * FLOAT32_BYTES,
        "interpolated_output_bytes": queries * channels * FLOAT32_BYTES,
        "field_gradient_bytes": queries * dim * channels * FLOAT32_BYTES,
    }
    return _footprint_total(footprint)


def _leaf_level(num_leaves: int) -> int:
    return (num_leaves - 1).bit_length()


def _real_nodes_at_level(num_leaves: int, level: int) -> int:
    leaf_level = _leaf_level(num_leaves)
    virtual_leaves = (1 << leaf_level) - num_leaves
    return (1 << level) - (virtual_leaves >> (leaf_level - level))


def _displaced_query_tensor_footprint(
    *,
    batch_size: int,
    n: int,
    heads: int,
    k: int,
    dim: int,
    channels: int,
) -> dict[str, int]:
    flat_queries = n * heads
    query_neighbors = batch_size * flat_queries * k
    footprint = {
        "query_output_indices_bytes": query_neighbors * INT64_BYTES,
        "query_output_squared_distances_bytes": query_neighbors * FLOAT32_BYTES,
        "source_position_gather_bytes": query_neighbors * dim * FLOAT32_BYTES,
        "value_gather_bytes": query_neighbors * channels * FLOAT32_BYTES,
    }
    return _footprint_total(footprint)


def _cuda_memory_start(torch) -> dict[str, int]:
    torch.cuda.synchronize()
    start = {
        "cuda_memory_allocated_start_bytes": int(torch.cuda.memory_allocated()),
        "cuda_memory_reserved_start_bytes": int(torch.cuda.memory_reserved()),
    }
    torch.cuda.reset_peak_memory_stats()
    return start


def _cuda_memory_finish(torch, start: dict[str, int]) -> dict[str, int]:
    torch.cuda.synchronize()
    allocated_end = int(torch.cuda.memory_allocated())
    reserved_end = int(torch.cuda.memory_reserved())
    return {
        **start,
        "cuda_memory_allocated_end_bytes": allocated_end,
        "cuda_memory_allocated_delta_bytes": allocated_end - start["cuda_memory_allocated_start_bytes"],
        "cuda_memory_allocated_peak_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_memory_reserved_end_bytes": reserved_end,
        "cuda_memory_reserved_delta_bytes": reserved_end - start["cuda_memory_reserved_start_bytes"],
        "cuda_memory_reserved_peak_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _summarize_cuda_memory_samples(samples: list[dict[str, int]]) -> dict[str, int | str]:
    if not samples:
        return {
            "measurement_scope": "benchmark_case_timed_iteration",
            "sample_count": 0,
        }
    summary: dict[str, int | str] = {
        "measurement_scope": "benchmark_case_timed_iteration",
        "sample_count": len(samples),
    }
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        summary[f"{key}_min"] = min(values)
        summary[f"{key}_max"] = max(values)
    return summary


def _cuda_event_time_ms(torch, fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def _cuda_wall_time_ms(torch, fn) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def _time_cpu_ms(fn) -> tuple[float, Any]:
    start = time.perf_counter()
    result = fn()
    return (time.perf_counter() - start) * 1000.0, result


def _apply_distribution(torch, data, distribution: str, dim: int, generator):
    if distribution == "uniform":
        return data
    if distribution == "duplicate-heavy":
        duplicate_count = max(1, data.size(-2) // 4)
        data[..., :duplicate_count, :] = data[..., :1, :]
        return data
    if distribution == "degenerate-axis":
        data[..., -1] = 0.5
        return data
    if distribution == "planar-3d":
        if dim == 3:
            data[..., 2] = 0.0
        else:
            data[..., -1] = 0.0
        return data
    if distribution == "collinear-3d":
        t = torch.linspace(0.0, 1.0, data.size(-2), device=data.device, dtype=data.dtype)
        line = torch.stack([t, 0.5 * t], dim=-1) if dim == 2 else torch.stack([t, 0.5 * t, -t], dim=-1)
        data.copy_(line.expand_as(data))
        return data
    if distribution == "clustered":
        centers = torch.rand((*data.shape[:-2], 4, dim), device=data.device, dtype=data.dtype, generator=generator)
        labels = torch.randint(0, 4, data.shape[:-1], device=data.device, generator=generator)
        expanded_centers = centers.gather(
            -2,
            labels.unsqueeze(-1).expand(*labels.shape, dim),
        )
        noise = torch.randn(data.shape, device=data.device, dtype=data.dtype, generator=generator)
        data.copy_((expanded_centers + 0.025 * noise).clamp_(0.0, 1.0))
        return data
    if distribution == "outlier-heavy":
        outlier_count = max(1, data.size(-2) // 20)
        data[..., :outlier_count, :] = 10.0 + torch.rand(
            (*data.shape[:-2], outlier_count, dim),
            device=data.device,
            dtype=data.dtype,
            generator=generator,
        )
        return data
    raise ValueError(f"unsupported distribution: {distribution}")


def _make_inputs(torch, n: int, queries: int, dim: int, features: int, seed: int, distribution: str):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + 100 * dim + n + queries)
    points = torch.rand((n, dim), device="cuda", dtype=torch.float32, generator=generator).contiguous()
    query_points = torch.rand((queries, dim), device="cuda", dtype=torch.float32, generator=generator).contiguous()
    _apply_distribution(torch, points, distribution, dim, generator)
    _apply_distribution(torch, query_points, distribution, dim, generator)
    feats = torch.rand((n, features), device="cuda", dtype=torch.float32, generator=generator).contiguous()
    return points, query_points, feats


def _make_batched_inputs(
    torch,
    batch_size: int,
    n: int,
    queries: int,
    dim: int,
    features: int,
    seed: int,
    distribution: str,
):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + 10_000 * batch_size + 100 * dim + n + queries)
    points = torch.rand((batch_size, n, dim), device="cuda", dtype=torch.float32, generator=generator).contiguous()
    query_points = torch.rand(
        (batch_size, queries, dim),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ).contiguous()
    _apply_distribution(torch, points, distribution, dim, generator)
    _apply_distribution(torch, query_points, distribution, dim, generator)
    feats = torch.rand(
        (batch_size, n, features),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ).contiguous()
    return points, query_points, feats


def _make_displaced_query_inputs(
    torch,
    batch_size: int,
    n: int,
    dim: int,
    heads: int,
    channels: int,
    seed: int,
    distribution: str,
):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + 20_000 * batch_size + 100 * dim + n + heads + channels)
    pos = torch.rand((batch_size, n, dim), device="cuda", dtype=torch.float32, generator=generator).contiguous()
    _apply_distribution(torch, pos, distribution, dim, generator)
    rho = 0.025 * torch.randn((batch_size, n, heads, dim), device="cuda", dtype=torch.float32, generator=generator)
    q = (pos[:, :, None, :] + rho).contiguous()
    values = torch.rand((batch_size, n, heads, channels), device="cuda", dtype=torch.float32, generator=generator)
    return pos, q, values.contiguous()


def _make_ragged_inputs(torch, ragged_n: list[int], ragged_queries: list[int], dim: int, features: int, seed: int, distribution: str):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + 30_000 + 100 * dim + sum(ragged_n) + sum(ragged_queries))
    point_samples = []
    query_samples = []
    feature_samples = []
    for sample, (n, queries) in enumerate(zip(ragged_n, ragged_queries)):
        points = torch.rand((n, dim), device="cuda", dtype=torch.float32, generator=generator).contiguous()
        query_points = torch.rand((queries, dim), device="cuda", dtype=torch.float32, generator=generator).contiguous()
        _apply_distribution(torch, points, distribution, dim, generator)
        _apply_distribution(torch, query_points, distribution, dim, generator)
        if sample:
            shift = torch.zeros((dim,), device="cuda", dtype=torch.float32)
            shift[0] = float(sample) * 3.0
            points += shift
            query_points += shift
        point_samples.append(points)
        query_samples.append(query_points)
        feature_samples.append(torch.rand((n, features), device="cuda", dtype=torch.float32, generator=generator))
    points = torch.cat(point_samples, dim=0).contiguous()
    query_points = torch.cat(query_samples, dim=0).contiguous()
    features = torch.cat(feature_samples, dim=0).contiguous()
    point_offsets = torch.tensor([0, *torch.tensor(ragged_n, device="cpu").cumsum(0).tolist()], device="cuda", dtype=torch.int64)
    query_offsets = torch.tensor(
        [0, *torch.tensor(ragged_queries, device="cpu").cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int64,
    )
    return points, query_points, features, point_offsets, query_offsets


def _run_batched_gpu_case(
    torch,
    torchbvh,
    points,
    query_points,
    features,
    k: int,
    include_mls: bool,
):
    timings: dict[str, list[float]] = {
        "native_batched_build_ms": [],
        "native_batched_query_traversal_ms": [],
        "native_batched_position_gather_ms": [],
        "native_batched_query_ms": [],
        "native_batched_total_path_ms": [],
        "single_sample_loop_total_path_ms": [],
    }
    if include_mls:
        timings["native_batched_total_with_mls_ms"] = []
        timings["single_sample_loop_total_with_mls_ms"] = []

    memory_start = _cuda_memory_start(torch)
    build_holder: dict[str, Any] = {}

    def build_once():
        build_holder["bvh"] = torchbvh.build_bvh_batched(points)

    timings["native_batched_build_ms"].append(_cuda_event_time_ms(torch, build_once))
    bvh = build_holder.pop("bvh")
    try:
        query_holder: dict[str, Any] = {}

        def query_once():
            query_holder["result"] = torchbvh.query_knn_batched(bvh, query_points, k)

        timings["native_batched_query_traversal_ms"].append(_cuda_event_time_ms(torch, query_once))
        indices, squared_distances = query_holder["result"]

        gather_holder: dict[str, Any] = {}

        def gather_once():
            gather_index = indices.unsqueeze(-1).expand(-1, -1, -1, int(points.size(-1)))
            expanded_source = points.unsqueeze(1).expand(-1, query_points.size(1), -1, -1)
            gather_holder["neighbor_positions"] = torch.gather(expanded_source, 2, gather_index)

        timings["native_batched_position_gather_ms"].append(_cuda_event_time_ms(torch, gather_once))
        neighbor_positions = gather_holder["neighbor_positions"]
        timings["native_batched_query_ms"].append(
            timings["native_batched_query_traversal_ms"][-1]
            + timings["native_batched_position_gather_ms"][-1]
        )

    finally:
        torchbvh.destroy_bvh(bvh)

    def native_total_once():
        total_bvh = torchbvh.build_bvh_batched(points)
        try:
            torchbvh.query_knn_batched(total_bvh, query_points, k, source_points=points)
        finally:
            torchbvh.destroy_bvh(total_bvh)

    timings["native_batched_total_path_ms"].append(_cuda_event_time_ms(torch, native_total_once))

    if include_mls:

        def native_total_with_mls_once():
            torchbvh.bvh_mls_interpolate_batched(points, query_points, features, k=k)

        timings["native_batched_total_with_mls_ms"].append(
            _cuda_event_time_ms(torch, native_total_with_mls_once)
        )

    def loop_total_once():
        for batch in range(points.size(0)):
            single = torchbvh.build_bvh(points[batch].contiguous())
            try:
                torchbvh.query_knn(
                    single,
                    query_points[batch].contiguous(),
                    k,
                    source_points=points[batch].contiguous(),
                )
            finally:
                torchbvh.destroy_bvh(single)

    timings["single_sample_loop_total_path_ms"].append(_cuda_event_time_ms(torch, loop_total_once))

    if include_mls:

        def loop_total_with_mls_once():
            for batch in range(points.size(0)):
                torchbvh.bvh_mls_interpolate(
                    points[batch].contiguous(),
                    query_points[batch].contiguous(),
                    features[batch].contiguous(),
                    k=k,
                )

        timings["single_sample_loop_total_with_mls_ms"].append(
            _cuda_event_time_ms(torch, loop_total_with_mls_once)
        )

    memory_stats = _cuda_memory_finish(torch, memory_start)
    return timings, memory_stats["cuda_memory_allocated_peak_bytes"], memory_stats


def _gather_displaced_query_values(torch, values, indices):
    batch_size, n, heads, channels = values.shape
    k = indices.size(-1)
    indices_bnhk = indices.reshape(batch_size, n, heads, k).permute(0, 2, 1, 3)
    values_bhnc = values.permute(0, 2, 1, 3)
    batch_index = torch.arange(batch_size, device=values.device).view(batch_size, 1, 1, 1)
    head_index = torch.arange(heads, device=values.device).view(1, heads, 1, 1)
    gathered = values_bhnc[batch_index, head_index, indices_bnhk, :]
    return gathered.permute(0, 2, 1, 3, 4).contiguous()


def _run_displaced_query_gpu_case(torch, torchbvh, pos, q, values, k: int):
    batch_size, n, heads, dim = q.shape
    flat_queries = q.reshape(batch_size, n * heads, dim).contiguous()
    timings: dict[str, list[float]] = {
        "displaced_query_batched_build_ms": [],
        "displaced_query_traversal_ms": [],
        "displaced_query_position_gather_ms": [],
        "displaced_query_ms": [],
        "displaced_query_value_gather_ms": [],
        "displaced_query_interpolate_ms": [],
        "displaced_query_total_path_ms": [],
    }
    memory_start = _cuda_memory_start(torch)
    build_holder: dict[str, Any] = {}

    def build_once():
        build_holder["bvh"] = torchbvh.build_bvh_batched(pos)

    timings["displaced_query_batched_build_ms"].append(_cuda_event_time_ms(torch, build_once))
    bvh = build_holder.pop("bvh")
    try:
        query_holder: dict[str, Any] = {}

        def query_once():
            query_holder["result"] = torchbvh.query_knn_batched(bvh, flat_queries, k)

        timings["displaced_query_traversal_ms"].append(_cuda_event_time_ms(torch, query_once))
        indices, _squared_distances = query_holder["result"]

        def position_gather_once():
            gather_index = indices.unsqueeze(-1).expand(-1, -1, -1, dim)
            expanded_source = pos.unsqueeze(1).expand(-1, flat_queries.size(1), -1, -1)
            torch.gather(expanded_source, 2, gather_index)

        timings["displaced_query_position_gather_ms"].append(_cuda_event_time_ms(torch, position_gather_once))
        timings["displaced_query_ms"].append(
            timings["displaced_query_traversal_ms"][-1] + timings["displaced_query_position_gather_ms"][-1]
        )

        def value_gather_once():
            public_indices = indices.reshape(batch_size, n, heads, k).contiguous()
            torchbvh.gather_neighbor_values(values, public_indices)

        timings["displaced_query_value_gather_ms"].append(_cuda_event_time_ms(torch, value_gather_once))

        def interpolate_once():
            torchbvh.interpolate_displaced(pos, q, values, k)

        timings["displaced_query_interpolate_ms"].append(_cuda_event_time_ms(torch, interpolate_once))
    finally:
        torchbvh.destroy_bvh(bvh)

    def total_once():
        total_bvh = torchbvh.build_bvh_batched(pos)
        try:
            indices, _squared_distances = torchbvh.query_knn_batched(total_bvh, flat_queries, k)
            gather_index = indices.unsqueeze(-1).expand(-1, -1, -1, dim)
            expanded_source = pos.unsqueeze(1).expand(-1, flat_queries.size(1), -1, -1)
            torch.gather(expanded_source, 2, gather_index)
            public_indices = indices.reshape(batch_size, n, heads, k).contiguous()
            torchbvh.gather_neighbor_values(values, public_indices)
        finally:
            torchbvh.destroy_bvh(total_bvh)

    timings["displaced_query_total_path_ms"].append(_cuda_event_time_ms(torch, total_once))
    memory_stats = _cuda_memory_finish(torch, memory_start)
    return timings, memory_stats["cuda_memory_allocated_peak_bytes"], memory_stats


def _ragged_gather_positions(torch, points, point_offsets, query_offsets, indices):
    gathered = []
    point_offsets_cpu = [int(v) for v in point_offsets.detach().cpu().tolist()]
    query_offsets_cpu = [int(v) for v in query_offsets.detach().cpu().tolist()]
    for batch in range(len(point_offsets_cpu) - 1):
        p_start, p_end = point_offsets_cpu[batch], point_offsets_cpu[batch + 1]
        q_start, q_end = query_offsets_cpu[batch], query_offsets_cpu[batch + 1]
        gathered.append(points[p_start:p_end][indices[q_start:q_end]])
    return torch.cat(gathered, dim=0)


def _run_ragged_gpu_case(torch, torchbvh, points, query_points, point_offsets, query_offsets, k: int):
    timings: dict[str, list[float]] = {
        "ragged_build_ms": [],
        "ragged_query_traversal_ms": [],
        "ragged_position_gather_ms": [],
        "ragged_query_ms": [],
        "ragged_total_path_ms": [],
    }
    memory_start = _cuda_memory_start(torch)
    build_holder: dict[str, Any] = {}

    def build_once():
        build_holder["bvh"] = torchbvh.build_bvh_ragged(points, point_offsets)

    timings["ragged_build_ms"].append(_cuda_event_time_ms(torch, build_once))
    bvh = build_holder.pop("bvh")
    try:
        query_holder: dict[str, Any] = {}

        def query_once():
            query_holder["result"] = torchbvh.query_knn_ragged(bvh, query_points, query_offsets, k)

        timings["ragged_query_traversal_ms"].append(_cuda_event_time_ms(torch, query_once))
        indices, _squared_distances = query_holder["result"]

        def gather_once():
            _ragged_gather_positions(torch, points, point_offsets, query_offsets, indices)

        timings["ragged_position_gather_ms"].append(_cuda_event_time_ms(torch, gather_once))
        timings["ragged_query_ms"].append(
            timings["ragged_query_traversal_ms"][-1] + timings["ragged_position_gather_ms"][-1]
        )
    finally:
        torchbvh.destroy_bvh(bvh)

    def total_once():
        total_bvh = torchbvh.build_bvh_ragged(points, point_offsets)
        try:
            indices, _squared_distances = torchbvh.query_knn_ragged(
                total_bvh,
                query_points,
                query_offsets,
                k,
            )
            _ragged_gather_positions(torch, points, point_offsets, query_offsets, indices)
        finally:
            torchbvh.destroy_bvh(total_bvh)

    timings["ragged_total_path_ms"].append(_cuda_event_time_ms(torch, total_once))
    memory_stats = _cuda_memory_finish(torch, memory_start)
    return timings, memory_stats["cuda_memory_allocated_peak_bytes"], memory_stats


def _run_gpu_case(
    torch,
    torchbvh,
    points,
    query_points,
    features,
    k: int,
    include_mls: bool,
):
    timings: dict[str, list[float]] = {
        "gpu_bvh_build_ms": [],
        "gpu_bvh_query_traversal_ms": [],
        "gpu_position_gather_ms": [],
        "gpu_bvh_query_ms": [],
        "gpu_total_bvh_path_ms": [],
    }
    if include_mls:
        timings["gpu_total_with_mls_ms"] = []

    memory_start = _cuda_memory_start(torch)
    build_holder: dict[str, Any] = {}

    def build_once():
        build_holder["bvh"] = torchbvh.build_bvh(points)

    timings["gpu_bvh_build_ms"].append(_cuda_event_time_ms(torch, build_once))
    bvh = build_holder.pop("bvh")
    try:
        query_holder: dict[str, Any] = {}

        def query_once():
            query_holder["result"] = torchbvh.query_knn(
                bvh,
                query_points,
                k,
            )

        timings["gpu_bvh_query_traversal_ms"].append(_cuda_event_time_ms(torch, query_once))
        indices, squared_distances = query_holder["result"]

        gather_holder: dict[str, Any] = {}

        def gather_once():
            gather_holder["neighbor_positions"] = points[indices]

        timings["gpu_position_gather_ms"].append(_cuda_event_time_ms(torch, gather_once))
        neighbor_positions = gather_holder["neighbor_positions"]
        timings["gpu_bvh_query_ms"].append(
            timings["gpu_bvh_query_traversal_ms"][-1] + timings["gpu_position_gather_ms"][-1]
        )

    finally:
        torchbvh.destroy_bvh(bvh)

    def total_once():
        bvh_total = torchbvh.build_bvh(points)
        try:
            torchbvh.query_knn(
                bvh_total,
                query_points,
                k,
                source_points=points,
            )
        finally:
            torchbvh.destroy_bvh(bvh_total)

    timings["gpu_total_bvh_path_ms"].append(_cuda_event_time_ms(torch, total_once))

    if include_mls:

        def total_with_mls_once():
            torchbvh.bvh_mls_interpolate(points, query_points, features, k=k)

        timings["gpu_total_with_mls_ms"].append(_cuda_event_time_ms(torch, total_with_mls_once))
    memory_stats = _cuda_memory_finish(torch, memory_start)
    return timings, memory_stats["cuda_memory_allocated_peak_bytes"], memory_stats


def _run_cpu_case(torch, cKDTree, points, query_points, k: int):
    timings: dict[str, list[float]] = {
        "cpu_transfer_to_host_ms": [],
        "cpu_kdtree_build_ms": [],
        "cpu_kdtree_query_ms": [],
        "cpu_transfer_to_device_ms": [],
        "cpu_total_kdtree_path_ms": [],
    }

    def to_host():
        torch.cuda.synchronize()
        points_cpu = points.detach().cpu().numpy()
        queries_cpu = query_points.detach().cpu().numpy()
        return points_cpu, queries_cpu

    timings["cpu_transfer_to_host_ms"].append(_time_cpu_ms(to_host)[0])
    points_cpu, queries_cpu = to_host()

    build_ms, tree = _time_cpu_ms(lambda: cKDTree(points_cpu))
    timings["cpu_kdtree_build_ms"].append(build_ms)

    query_ms, query_result = _time_cpu_ms(lambda: tree.query(queries_cpu, k=k))
    timings["cpu_kdtree_query_ms"].append(query_ms)
    distances, indices = query_result

    def to_device():
        torch.as_tensor(indices, device=points.device, dtype=torch.int64)
        torch.as_tensor(distances, device=points.device, dtype=torch.float32).square()
        torch.cuda.synchronize()

    timings["cpu_transfer_to_device_ms"].append(_time_cpu_ms(to_device)[0])

    def total_once():
        points_cpu_total, queries_cpu_total = to_host()
        tree_total = cKDTree(points_cpu_total)
        distances_total, indices_total = tree_total.query(queries_cpu_total, k=k)
        torch.as_tensor(indices_total, device=points.device, dtype=torch.int64)
        torch.as_tensor(distances_total, device=points.device, dtype=torch.float32).square()
        torch.cuda.synchronize()

    timings["cpu_total_kdtree_path_ms"].append(_time_cpu_ms(total_once)[0])
    return timings


def _torch_cluster_batch_vector(torch, batch_size: int, count: int):
    return torch.arange(batch_size, device="cuda", dtype=torch.long).repeat_interleave(count)


def _torch_cluster_dense_conversion(torch, edge_index, batch_size: int, queries_per_sample: int, n: int, k: int):
    query_global = edge_index[0].long()
    source_global = edge_index[1].long()
    query_batch = query_global // queries_per_sample
    query_local = query_global - query_batch * queries_per_sample
    source_local = source_global - query_batch * n
    rank = torch.arange(edge_index.size(1), device=edge_index.device, dtype=torch.long) % k
    dense = torch.empty((batch_size, queries_per_sample, k), device=edge_index.device, dtype=torch.long)
    dense[query_batch, query_local, rank] = source_local
    return dense


def _run_torch_cluster_case(torch, torch_cluster_knn, points, query_points, k: int):
    if points.dim() == 2:
        batch_size = 1
        n = int(points.size(0))
        queries_per_sample = int(query_points.size(0))
        flat_points = points.contiguous()
        flat_queries = query_points.contiguous()
    else:
        batch_size = int(points.size(0))
        n = int(points.size(1))
        queries_per_sample = int(query_points.size(1))
        flat_points = points.reshape(batch_size * n, points.size(-1)).contiguous()
        flat_queries = query_points.reshape(batch_size * queries_per_sample, query_points.size(-1)).contiguous()

    batch_points = _torch_cluster_batch_vector(torch, batch_size, n)
    batch_queries = _torch_cluster_batch_vector(torch, batch_size, queries_per_sample)
    memory_start = _cuda_memory_start(torch)
    timings: dict[str, list[float]] = {
        "torch_cluster_knn_ms": [],
        "torch_cluster_dense_conversion_ms": [],
        "torch_cluster_total_ms": [],
    }
    edge_holder: dict[str, Any] = {}

    def knn_once():
        edge_holder["edge_index"] = torch_cluster_knn(flat_points, flat_queries, k, batch_points, batch_queries)

    timings["torch_cluster_knn_ms"].append(_cuda_event_time_ms(torch, knn_once))
    edge_index = edge_holder["edge_index"]

    def conversion_once():
        _torch_cluster_dense_conversion(torch, edge_index, batch_size, queries_per_sample, n, k)

    timings["torch_cluster_dense_conversion_ms"].append(_cuda_event_time_ms(torch, conversion_once))
    timings["torch_cluster_total_ms"].append(
        timings["torch_cluster_knn_ms"][-1] + timings["torch_cluster_dense_conversion_ms"][-1]
    )
    memory_stats = _cuda_memory_finish(torch, memory_start)
    return timings, {
        "package": "torch_cluster",
        "package_version": getattr(sys.modules.get("torch_cluster"), "__version__", None),
        "function": "torch_cluster.knn",
        "output_shape": list(edge_index.shape),
        "dense_conversion_shape": [batch_size, queries_per_sample, k],
        "comparison_scope": "knn_core_plus_dense_indices_conversion",
        "cuda_allocator_memory": memory_stats,
        "peak_gpu_memory_bytes": memory_stats["cuda_memory_allocated_peak_bytes"],
        "conversion_note": (
            "torch_cluster.knn returns edge-index pairs; dense conversion reshapes "
            "query-major global indices into local (B, M, k) neighbor indices."
        ),
    }


def _cupy_knn_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "leaf_size": args.cupy_knn_leaf_size,
        "compact": args.cupy_knn_compact,
        "shrink_to_fit": args.cupy_knn_shrink_to_fit,
        "sort_queries": args.cupy_knn_sort_queries,
    }


def _cupy_knn_option_variants(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    base = _cupy_knn_options(args)
    if args.cupy_knn_sort_mode == "both":
        sorted_options = dict(base)
        sorted_options["sort_queries"] = True
        unsorted_options = dict(base)
        unsorted_options["sort_queries"] = False
        return [
            ("sorted", sorted_options),
            ("unsorted", unsorted_options),
        ]
    if args.cupy_knn_sort_mode == "sorted":
        options = dict(base)
        options["sort_queries"] = True
        return [("sorted", options)]
    if args.cupy_knn_sort_mode == "unsorted":
        options = dict(base)
        options["sort_queries"] = False
        return [("unsorted", options)]
    return [("sorted" if base["sort_queries"] else "unsorted", base)]


def _load_cupy_knn_baseline():
    try:
        import cupy as cp
    except ImportError as exc:
        raise SystemExit(
            "cupy is required for --cupy-knn-baseline. Install a CUDA-matched package "
            "such as cupy-cuda12x, or install cupy-knn[cuda12x]."
        ) from exc

    try:
        from cupy_knn import LBVHIndex
        import cupy_knn.lbvh_index as lbvh_module
    except ImportError:
        vendored = Path(__file__).resolve().parents[1] / "cupy-knn-master"
        if vendored.exists() and str(vendored) not in sys.path:
            sys.path.insert(0, str(vendored))
        try:
            from cupy_knn import LBVHIndex
            import cupy_knn.lbvh_index as lbvh_module
        except ImportError as exc:
            raise SystemExit(
                "cupy_knn is required for --cupy-knn-baseline. Install it or make "
                f"the vendored package importable from {vendored}."
            ) from exc

    _patch_cupy_knn_compile_flags(cp, lbvh_module)
    return cp, LBVHIndex


def _patch_cupy_knn_compile_flags(cp, lbvh_module) -> None:
    flags = tuple(getattr(lbvh_module, "_compile_flags", ()))
    patched_flags = tuple("--std=c++17" if flag == "--std=c++11" else flag for flag in flags)
    if patched_flags == flags:
        return
    lbvh_module._compile_flags = patched_flags
    lbvh_module._construct_tree_kernels = cp.RawModule(
        code=lbvh_module._lbvh_src,
        options=patched_flags,
        name_expressions=(
            "compute_morton_kernel",
            "compute_morton_points_kernel",
            "initialize_tree_kernel",
            "construct_tree_kernel",
            "optimize_tree_kernel",
            "compute_free_indices_kernel",
            "compact_tree_kernel",
        ),
    )


def _torch_to_cupy(torch, cp, tensor):
    if tensor.dtype != torch.float32 or not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError("_torch_to_cupy expects a contiguous CUDA float32 tensor")
    return cp.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))


def _cupy_wall_time_ms(torch, cp, fn) -> float:
    torch.cuda.synchronize()
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    fn()
    cp.cuda.Stream.null.synchronize()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def _cupy_knn_sample_shape(points, query_points) -> dict[str, int | list[int] | str]:
    if points.dim() == 2:
        return {
            "batch_mode": "single",
            "batch_size": 1,
            "points_shape": list(points.shape),
            "query_shape": list(query_points.shape),
            "queries_per_sample": int(query_points.size(0)),
        }
    return {
        "batch_mode": "per_sample_loop",
        "batch_size": int(points.size(0)),
        "points_shape": list(points.shape),
        "query_shape": list(query_points.shape),
        "queries_per_sample": int(query_points.size(1)),
    }


def _cupy_to_torch_float32(torch, cp, array):
    return torch.utils.dlpack.from_dlpack(array.astype(cp.float32, copy=False))


def _cupy_distance_correctness(torch, cp, cupy_distances, reference_squared_distances) -> dict[str, float | None]:
    if reference_squared_distances is None or cupy_distances is None:
        return {
            "max_abs_distance_error": None,
            "max_rel_distance_error": None,
        }
    distances = _cupy_to_torch_float32(torch, cp, cupy_distances).reshape_as(reference_squared_distances)
    sorted_distances = distances.sort(dim=-1).values
    sorted_reference = reference_squared_distances.detach().sort(dim=-1).values
    abs_error = (sorted_distances - sorted_reference).abs()
    rel_error = abs_error / sorted_reference.abs().clamp_min(1.0e-12)
    return {
        "max_abs_distance_error": float(abs_error.max().item()),
        "max_rel_distance_error": float(rel_error.max().item()),
    }


def _implicit_reference_distances(torch, torchbvh, points, query_points, k: int):
    if points.dim() == 2:
        bvh = torchbvh.build_bvh(points)
        try:
            _indices, distances = torchbvh.query_knn(bvh, query_points, k)
            return distances.detach()
        finally:
            torchbvh.destroy_bvh(bvh)
    bvh = torchbvh.build_bvh_batched(points)
    try:
        _indices, distances = torchbvh.query_knn_batched(bvh, query_points, k)
        return distances.detach()
    finally:
        torchbvh.destroy_bvh(bvh)


def _run_cupy_knn_case(
    torch,
    cp,
    LBVHIndex,
    points,
    query_points,
    k: int,
    options: dict[str, Any],
    reference_squared_distances=None,
):
    if points.size(-1) != 3 or query_points.size(-1) != 3:
        raise ValueError("_run_cupy_knn_case only supports 3D point tensors")

    timings: dict[str, list[float]] = {
        "cupy_knn_build_ms": [],
        "cupy_knn_prepare_ms": [],
        "cupy_knn_query_ms": [],
        "cupy_knn_total_build_prepare_query_ms": [],
    }
    memory_start = _cuda_memory_start(torch)
    shape_metadata = _cupy_knn_sample_shape(points, query_points)

    if points.dim() == 2:
        point_samples = [points]
        query_samples = [query_points]
    else:
        point_samples = [points[batch].contiguous() for batch in range(points.size(0))]
        query_samples = [query_points[batch].contiguous() for batch in range(query_points.size(0))]

    query_distance_outputs = []
    query_index_outputs = []
    query_count_outputs = []

    build_total = 0.0
    prepare_total = 0.0
    query_total = 0.0
    total_path = 0.0
    for sample_points, sample_queries in zip(point_samples, query_samples):
        points_cp = _torch_to_cupy(torch, cp, sample_points)
        queries_cp = _torch_to_cupy(torch, cp, sample_queries)
        holder: dict[str, Any] = {}

        def build_once():
            index = LBVHIndex(**options)
            index.build(points_cp)
            holder["index"] = index

        build_total += _cupy_wall_time_ms(torch, cp, build_once)
        index = holder["index"]

        def prepare_once():
            index.prepare_knn_default(k)

        prepare_total += _cupy_wall_time_ms(torch, cp, prepare_once)

        def query_once():
            holder["query_result"] = index.query_knn(queries_cp)

        query_total += _cupy_wall_time_ms(torch, cp, query_once)
        indices_out, distances_out, counts_out = holder["query_result"]
        query_index_outputs.append(indices_out)
        query_distance_outputs.append(distances_out)
        query_count_outputs.append(counts_out)

        def total_once():
            total_index = LBVHIndex(**options)
            total_index.build(points_cp)
            total_index.prepare_knn_default(k)
            total_index.query_knn(queries_cp)

        total_path += _cupy_wall_time_ms(torch, cp, total_once)

    timings["cupy_knn_build_ms"].append(build_total)
    timings["cupy_knn_prepare_ms"].append(prepare_total)
    timings["cupy_knn_query_ms"].append(query_total)
    timings["cupy_knn_total_build_prepare_query_ms"].append(total_path)

    if points.dim() == 2:
        distances_for_check = query_distance_outputs[0]
        output_shape = list(query_distance_outputs[0].shape)
        count_shape = list(query_count_outputs[0].shape)
    else:
        distances_for_check = cp.stack(query_distance_outputs, axis=0)
        output_shape = list(distances_for_check.shape)
        count_shape = [len(query_count_outputs), *query_count_outputs[0].shape]

    correctness = _cupy_distance_correctness(
        torch,
        cp,
        distances_for_check,
        reference_squared_distances,
    )
    memory_stats = _cuda_memory_finish(torch, memory_start)
    metadata = {
        "package": "cupy_knn",
        "package_version": getattr(sys.modules.get("cupy_knn"), "__version__", None),
        "index_class": "LBVHIndex",
        "comparison_scope": "knn_core_build_prepare_query",
        "output_indices_dtype": str(query_index_outputs[0].dtype),
        "output_distances_dtype": str(query_distance_outputs[0].dtype),
        "output_counts_dtype": str(query_count_outputs[0].dtype),
        "output_indices_shape": output_shape,
        "output_distances_shape": output_shape,
        "output_counts_shape": count_shape,
        "options": dict(options),
        "batch_handling": shape_metadata["batch_mode"],
        "shape": shape_metadata,
        "cuda_allocator_memory": memory_stats,
        "peak_gpu_memory_bytes": memory_stats["cuda_memory_allocated_peak_bytes"],
        "correctness": correctness,
        "notes": [
            "cupy_knn supports 3D inputs only",
            "fixed-size batched and displaced-query modes run cupy_knn once per sample",
            "prepare_knn_default is timed separately because it prepares or compiles the k-specific query kernel",
        ],
    }
    return timings, metadata


def _prefix_timing_keys(timings: dict[str, list[float]], prefix: str) -> dict[str, list[float]]:
    return {f"{prefix}_{key}": values for key, values in timings.items()}


def _run_cupy_knn_variants(
    torch,
    cp,
    LBVHIndex,
    points,
    query_points,
    k: int,
    variants: list[tuple[str, dict[str, Any]]],
    reference_squared_distances=None,
):
    merged_timings: dict[str, list[float]] = {}
    metadata: dict[str, Any] = {}
    for label, options in variants:
        timings, variant_metadata = _run_cupy_knn_case(
            torch,
            cp,
            LBVHIndex,
            points,
            query_points,
            k,
            options,
            reference_squared_distances=reference_squared_distances,
        )
        if len(variants) == 1:
            _merge_timings(merged_timings, timings)
            metadata = variant_metadata
        else:
            _merge_timings(merged_timings, _prefix_timing_keys(timings, f"cupy_knn_{label}"))
            metadata[label] = variant_metadata
    return merged_timings, metadata


def _bvh_scene_bounds_for_queries(torch, data: Mapping, query_points):
    if query_points.dim() == 2:
        return (
            query_points.unsqueeze(0),
            data["scene_min"].unsqueeze(0),
            data["scene_max"].unsqueeze(0),
        )
    return query_points, data["scene_min"], data["scene_max"]


def _run_gpu_sorted_query_case(torch, torchbvh, points, query_points, k: int):
    from torchbvh._reorder import morton_sort_queries_batched

    timings: dict[str, list[float]] = {
        "gpu_sorted_bvh_build_ms": [],
        "gpu_sorted_query_sort_ms": [],
        "gpu_sorted_query_traversal_ms": [],
        "gpu_sorted_output_unsort_ms": [],
        "gpu_sorted_position_gather_ms": [],
        "gpu_sorted_bvh_query_ms": [],
        "gpu_sorted_total_bvh_path_ms": [],
    }
    memory_start = _cuda_memory_start(torch)
    build_holder: dict[str, Any] = {}

    if points.dim() == 2:
        build_fn = torchbvh.build_bvh
    else:
        build_fn = torchbvh.build_bvh_batched

    def build_once():
        build_holder["bvh"] = build_fn(points)

    timings["gpu_sorted_bvh_build_ms"].append(_cuda_event_time_ms(torch, build_once))
    bvh = build_holder.pop("bvh")
    try:
        data = bvh._require_live() if hasattr(bvh, "_require_live") else bvh
        query_batch, scene_min, scene_max = _bvh_scene_bounds_for_queries(torch, data, query_points)
        sort_holder: dict[str, Any] = {}

        def sort_once():
            sort_perm, inv_perm = morton_sort_queries_batched(query_batch, scene_min, scene_max)
            sort_holder["sort_perm"] = sort_perm
            sort_holder["inv_perm"] = inv_perm

        timings["gpu_sorted_query_sort_ms"].append(_cuda_event_time_ms(torch, sort_once))
        sort_perm = sort_holder["sort_perm"]

        query_holder: dict[str, Any] = {}

        def query_once():
            if query_points.dim() == 2:
                query_holder["result"] = torchbvh._C.query_knn_ordered(
                    data["node_aabbs"],
                    data["sorted_indices"],
                    query_points,
                    sort_perm.squeeze(0).contiguous(),
                    data["num_leaves"],
                    data["leaf_level"],
                    data["dim"],
                    k,
                )
            else:
                query_holder["result"] = torchbvh._C.query_knn_batched_ordered(
                    data["node_aabbs"],
                    data["sorted_indices"],
                    query_points,
                    sort_perm.contiguous(),
                    data["num_leaves"],
                    data["num_real_nodes"],
                    data["leaf_level"],
                    data["dim"],
                    k,
                )

        timings["gpu_sorted_query_traversal_ms"].append(_cuda_event_time_ms(torch, query_once))
        indices, _squared_distances = query_holder["result"]
        timings["gpu_sorted_output_unsort_ms"].append(0.0)

        def gather_once():
            if points.dim() == 2:
                points[indices]
            else:
                gather_index = indices.unsqueeze(-1).expand(-1, -1, -1, int(points.size(-1)))
                expanded_source = points.unsqueeze(1).expand(-1, query_points.size(1), -1, -1)
                torch.gather(expanded_source, 2, gather_index)

        timings["gpu_sorted_position_gather_ms"].append(_cuda_event_time_ms(torch, gather_once))
        timings["gpu_sorted_bvh_query_ms"].append(
            timings["gpu_sorted_query_sort_ms"][-1]
            + timings["gpu_sorted_query_traversal_ms"][-1]
            + timings["gpu_sorted_output_unsort_ms"][-1]
            + timings["gpu_sorted_position_gather_ms"][-1]
        )
    finally:
        torchbvh.destroy_bvh(bvh)

    def total_once():
        total_bvh = build_fn(points)
        try:
            data = total_bvh._require_live() if hasattr(total_bvh, "_require_live") else total_bvh
            query_batch, scene_min, scene_max = _bvh_scene_bounds_for_queries(torch, data, query_points)
            sort_perm, _inv_perm = morton_sort_queries_batched(query_batch, scene_min, scene_max)
            if query_points.dim() == 2:
                indices, _distances = torchbvh._C.query_knn_ordered(
                    data["node_aabbs"],
                    data["sorted_indices"],
                    query_points,
                    sort_perm.squeeze(0).contiguous(),
                    data["num_leaves"],
                    data["leaf_level"],
                    data["dim"],
                    k,
                )
            else:
                indices, _distances = torchbvh._C.query_knn_batched_ordered(
                    data["node_aabbs"],
                    data["sorted_indices"],
                    query_points,
                    sort_perm.contiguous(),
                    data["num_leaves"],
                    data["num_real_nodes"],
                    data["leaf_level"],
                    data["dim"],
                    k,
                )
            if points.dim() == 2:
                points[indices]
            else:
                gather_index = indices.unsqueeze(-1).expand(-1, -1, -1, int(points.size(-1)))
                expanded_source = points.unsqueeze(1).expand(-1, query_points.size(1), -1, -1)
                torch.gather(expanded_source, 2, gather_index)
        finally:
            torchbvh.destroy_bvh(total_bvh)

    timings["gpu_sorted_total_bvh_path_ms"].append(_cuda_event_time_ms(torch, total_once))
    memory_stats = _cuda_memory_finish(torch, memory_start)
    metadata = {
        "query_sort": "morton_sort_queries_batched",
        "output_order": "written_to_original_query_order_by_query_kernel",
        "timing_scope": "build_sort_ordered_query_position_gather",
        "cuda_allocator_memory": memory_stats,
        "peak_gpu_memory_bytes": memory_stats["cuda_memory_allocated_peak_bytes"],
    }
    return timings, memory_stats["cuda_memory_allocated_peak_bytes"], memory_stats, metadata


def _cupy_knn_speedups(summary: dict[str, dict[str, float | None]], package_total_key: str) -> dict[str, float | None]:
    package_total = summary[package_total_key]["mean_ms"]
    speedups: dict[str, float | None] = {}
    if "cupy_knn_total_build_prepare_query_ms" in summary:
        cupy_total = summary["cupy_knn_total_build_prepare_query_ms"]["mean_ms"]
        speedups["cupy_knn_total"] = cupy_total / package_total if package_total and cupy_total else math.nan
    for label in ("sorted", "unsorted"):
        key = f"cupy_knn_{label}_cupy_knn_total_build_prepare_query_ms"
        if key in summary:
            cupy_total = summary[key]["mean_ms"]
            speedups[f"cupy_knn_{label}_total"] = (
                cupy_total / package_total if package_total and cupy_total else math.nan
            )
    return speedups


def _merge_timings(dst: dict[str, list[float]], src: dict[str, list[float]]) -> None:
    for name, values in src.items():
        dst.setdefault(name, []).extend(values)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    import torchbvh

    if args.no_cpu_baseline:
        cKDTree = None
    else:
        try:
            from scipy.spatial import cKDTree
        except ImportError as exc:
            raise SystemExit("scipy is required for the CPU cKDTree baseline") from exc
    if args.torch_cluster_baseline:
        try:
            from torch_cluster import knn as torch_cluster_knn
        except ImportError as exc:
            raise SystemExit("torch_cluster is required for --torch-cluster-baseline") from exc
    else:
        torch_cluster_knn = None
    if args.cupy_knn_baseline:
        cp, LBVHIndex = _load_cupy_knn_baseline()
        cupy_knn_variants = _cupy_knn_option_variants(args)
    else:
        cp = None
        LBVHIndex = None
        cupy_knn_variants = None

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    cases = []
    include_mls = not args.no_mls
    if args.displaced_query and include_mls:
        include_mls = False
        displaced_query_notes = [
            "benchmark-local displaced-query mode reports build/query/source/value gathers only; multihead MLS is not measured here",
            "displaced_query_value_gather_ms uses the stable torchbvh.gather_neighbor_values helper on reshaped (B, N, H, k) indices",
            _query_count_config(args)["query_count_note"],
        ]
    else:
        displaced_query_notes = [
            "displaced_query_value_gather_ms uses the stable torchbvh.gather_neighbor_values helper on reshaped (B, N, H, k) indices",
            _query_count_config(args)["query_count_note"],
        ] if args.displaced_query else []
    if args.ragged_n is not None and include_mls:
        include_mls = False
        ragged_notes = [
            "ragged mode reports build/query/source gather only; ragged MLS is not measured here",
            "ragged benchmark uses the current Python-composed ragged API and local source gather timing",
            _query_count_config(args)["query_count_note"],
        ]
    else:
        ragged_notes = [
            "ragged benchmark uses the current Python-composed ragged API and local source gather timing",
            _query_count_config(args)["query_count_note"],
        ]
    for dim in args.dim:
        for k in args.k:
            if args.ragged_n is not None:
                points, query_points, _features, point_offsets, query_offsets = _make_ragged_inputs(
                    torch,
                    args.ragged_n,
                    args.ragged_queries,
                    dim,
                    args.features,
                    args.seed + 1000 * dim + k,
                    args.distribution,
                )
                for _ in range(args.warmup):
                    bvh = torchbvh.build_bvh_ragged(points, point_offsets)
                    try:
                        torchbvh.query_knn_ragged(bvh, query_points, query_offsets, k)
                    finally:
                        torchbvh.destroy_bvh(bvh)
                    torch.cuda.synchronize()

                samples: dict[str, list[float]] = {}
                memory_samples: list[dict[str, int]] = []
                peak_memory = 0
                for _ in range(args.iters):
                    timings, memory, memory_stats = _run_ragged_gpu_case(
                        torch,
                        torchbvh,
                        points,
                        query_points,
                        point_offsets,
                        query_offsets,
                        k,
                    )
                    _merge_timings(samples, timings)
                    memory_samples.append(memory_stats)
                    peak_memory = max(peak_memory, memory)
                cases.append(
                    {
                        "mode": "ragged",
                        **_stage8_case_metadata(args, "ragged"),
                        "resolution_geometry_policy": _resolution_geometry_policy("ragged"),
                        "ragged_n": args.ragged_n,
                        "ragged_queries": args.ragged_queries,
                        **_query_count_config(args),
                        "batch_size": len(args.ragged_n),
                        "total_n": sum(args.ragged_n),
                        "total_queries": sum(args.ragged_queries),
                        "dim": dim,
                        "k": k,
                        "features": args.features,
                        "distribution": args.distribution,
                        "include_mls": False,
                        "samples": samples,
                        "summary": summarize_timings(samples),
                        "peak_gpu_memory_bytes": peak_memory,
                        "cuda_allocator_memory": _summarize_cuda_memory_samples(memory_samples),
                        "tensor_footprint_estimates": {
                            "query_outputs": _query_output_footprint(
                                batch_size=1,
                                queries_per_sample=sum(args.ragged_queries),
                                k=k,
                                dim=dim,
                                include_source_position_gather=True,
                            )
                        },
                        "notes": ragged_notes,
                    }
                )
                continue

            if args.displaced_query:
                pos, q, values = _make_displaced_query_inputs(
                    torch,
                    args.batch_size,
                    args.n,
                    dim,
                    args.displaced_query_heads,
                    args.displaced_query_channels,
                    args.seed + 1000 * dim + k,
                    args.distribution,
                )
                flat_queries = q.reshape(args.batch_size, args.n * args.displaced_query_heads, dim).contiguous()
                for _ in range(args.warmup):
                    bvh = torchbvh.build_bvh_batched(pos)
                    try:
                        indices, _distances = torchbvh.query_knn_batched(bvh, flat_queries, k)
                        _gather_displaced_query_values(torch, values, indices)
                    finally:
                        torchbvh.destroy_bvh(bvh)
                    if torch_cluster_knn is not None:
                        _run_torch_cluster_case(torch, torch_cluster_knn, pos, flat_queries, k)
                    if args.gpu_sort_queries:
                        _run_gpu_sorted_query_case(torch, torchbvh, pos, flat_queries, k)
                    if cp is not None and LBVHIndex is not None and cupy_knn_variants is not None:
                        _run_cupy_knn_variants(torch, cp, LBVHIndex, pos, flat_queries, k, cupy_knn_variants)
                    torch.cuda.synchronize()

                samples = {}
                memory_samples: list[dict[str, int]] = []
                peak_memory = 0
                torch_cluster_metadata = None
                cupy_knn_metadata = None
                gpu_sorted_query_metadata = None
                cupy_reference_distances = None
                if cp is not None and LBVHIndex is not None and cupy_knn_variants is not None:
                    cupy_reference_distances = _implicit_reference_distances(torch, torchbvh, pos, flat_queries, k)
                for _ in range(args.iters):
                    timings, memory, memory_stats = _run_displaced_query_gpu_case(torch, torchbvh, pos, q, values, k)
                    _merge_timings(samples, timings)
                    if args.gpu_sort_queries:
                        timings, sorted_memory, sorted_memory_stats, gpu_sorted_query_metadata = _run_gpu_sorted_query_case(
                            torch,
                            torchbvh,
                            pos,
                            flat_queries,
                            k,
                        )
                        _merge_timings(samples, timings)
                        memory_samples.append(sorted_memory_stats)
                        peak_memory = max(peak_memory, sorted_memory)
                    if torch_cluster_knn is not None:
                        timings, torch_cluster_metadata = _run_torch_cluster_case(
                            torch,
                            torch_cluster_knn,
                            pos,
                            flat_queries,
                            k,
                        )
                        _merge_timings(samples, timings)
                    if cp is not None and LBVHIndex is not None and cupy_knn_variants is not None:
                        timings, cupy_knn_metadata = _run_cupy_knn_variants(
                            torch,
                            cp,
                            LBVHIndex,
                            pos,
                            flat_queries,
                            k,
                            cupy_knn_variants,
                            reference_squared_distances=cupy_reference_distances,
                        )
                        _merge_timings(samples, timings)
                    memory_samples.append(memory_stats)
                    peak_memory = max(peak_memory, memory)
                summary = summarize_timings(samples)
                torch_cluster_speedup = None
                if torch_cluster_metadata is not None:
                    package_total = summary["displaced_query_total_path_ms"]["mean_ms"]
                    torch_cluster_total = summary["torch_cluster_total_ms"]["mean_ms"]
                    torch_cluster_speedup = (
                        torch_cluster_total / package_total
                        if package_total and torch_cluster_total
                        else math.nan
                    )
                cupy_knn_speedups = _cupy_knn_speedups(summary, "displaced_query_total_path_ms")
                cupy_knn_speedup = cupy_knn_speedups.get("cupy_knn_total")
                cases.append(
                    {
                        "mode": "displaced_query",
                        **_stage8_case_metadata(args, "displaced_query"),
                        "resolution_geometry_policy": _resolution_geometry_policy("displaced_query"),
                        "batch_size": args.batch_size,
                        "n": args.n,
                        # Backward-compatible effective per-sample query count.
                        "queries": args.n * args.displaced_query_heads,
                        **_query_count_config(args),
                        "dim": dim,
                        "k": k,
                        "features": args.features,
                        "displaced_query_heads": args.displaced_query_heads,
                        "displaced_query_channels": args.displaced_query_channels,
                        "distribution": args.distribution,
                        "include_mls": False,
                        "samples": samples,
                        "summary": summary,
                        "torch_cluster_baseline": torch_cluster_metadata,
                        "torch_cluster_total_vs_displaced_total_speedup": torch_cluster_speedup,
                        "gpu_sorted_query_baseline": gpu_sorted_query_metadata,
                        "cupy_knn_baseline": cupy_knn_metadata,
                        "cupy_knn_total_vs_displaced_total_speedup": cupy_knn_speedup,
                        "cupy_knn_speedups_vs_displaced_total": cupy_knn_speedups,
                        "peak_gpu_memory_bytes": peak_memory,
                        "cuda_allocator_memory": _summarize_cuda_memory_samples(memory_samples),
                        "tensor_footprint_estimates": {
                            "displaced_query_outputs_and_gathers": _displaced_query_tensor_footprint(
                                batch_size=args.batch_size,
                                n=args.n,
                                heads=args.displaced_query_heads,
                                k=k,
                                dim=dim,
                                channels=args.displaced_query_channels,
                            )
                        },
                        "displaced_query_value_gather_helper": "torchbvh.gather_neighbor_values",
                        "notes": displaced_query_notes,
                    }
                )
                continue

            if args.batch_size > 1:
                points, query_points, features = _make_batched_inputs(
                    torch,
                    args.batch_size,
                    args.n,
                    args.queries,
                    dim,
                    args.features,
                    args.seed + 1000 * dim + k,
                    args.distribution,
                )
                for _ in range(args.warmup):
                    bvh = torchbvh.build_bvh_batched(points)
                    try:
                        torchbvh.query_knn_batched(bvh, query_points, k, source_points=points)
                        if include_mls:
                            torchbvh.bvh_mls_interpolate_batched(points, query_points, features, k=k)
                    finally:
                        torchbvh.destroy_bvh(bvh)
                    if torch_cluster_knn is not None:
                        _run_torch_cluster_case(torch, torch_cluster_knn, points, query_points, k)
                    if args.gpu_sort_queries:
                        _run_gpu_sorted_query_case(torch, torchbvh, points, query_points, k)
                    if cp is not None and LBVHIndex is not None and cupy_knn_variants is not None:
                        _run_cupy_knn_variants(torch, cp, LBVHIndex, points, query_points, k, cupy_knn_variants)
                    torch.cuda.synchronize()

                samples: dict[str, list[float]] = {}
                memory_samples: list[dict[str, int]] = []
                peak_memory = 0
                torch_cluster_metadata = None
                cupy_knn_metadata = None
                gpu_sorted_query_metadata = None
                cupy_reference_distances = None
                if cp is not None and LBVHIndex is not None and cupy_knn_variants is not None:
                    cupy_reference_distances = _implicit_reference_distances(torch, torchbvh, points, query_points, k)
                for _ in range(args.iters):
                    timings, memory, memory_stats = _run_batched_gpu_case(
                        torch,
                        torchbvh,
                        points,
                        query_points,
                        features,
                        k,
                        include_mls,
                    )
                    _merge_timings(samples, timings)
                    if args.gpu_sort_queries:
                        timings, sorted_memory, sorted_memory_stats, gpu_sorted_query_metadata = _run_gpu_sorted_query_case(
                            torch,
                            torchbvh,
                            points,
                            query_points,
                            k,
                        )
                        _merge_timings(samples, timings)
                        memory_samples.append(sorted_memory_stats)
                        peak_memory = max(peak_memory, sorted_memory)
                    if torch_cluster_knn is not None:
                        timings, torch_cluster_metadata = _run_torch_cluster_case(
                            torch,
                            torch_cluster_knn,
                            points,
                            query_points,
                            k,
                        )
                        _merge_timings(samples, timings)
                    if cp is not None and LBVHIndex is not None and cupy_knn_variants is not None:
                        timings, cupy_knn_metadata = _run_cupy_knn_variants(
                            torch,
                            cp,
                            LBVHIndex,
                            points,
                            query_points,
                            k,
                            cupy_knn_variants,
                            reference_squared_distances=cupy_reference_distances,
                        )
                        _merge_timings(samples, timings)
                    memory_samples.append(memory_stats)
                    peak_memory = max(peak_memory, memory)

                summary = summarize_timings(samples)
                native_total = summary["native_batched_total_path_ms"]["mean_ms"]
                loop_total = summary["single_sample_loop_total_path_ms"]["mean_ms"]
                speedup = (loop_total / native_total) if native_total and loop_total else math.nan
                torch_cluster_speedup = None
                if torch_cluster_metadata is not None:
                    torch_cluster_total = summary["torch_cluster_total_ms"]["mean_ms"]
                    torch_cluster_speedup = (
                        torch_cluster_total / native_total
                        if native_total and torch_cluster_total
                        else math.nan
                    )
                cupy_knn_speedups = _cupy_knn_speedups(summary, "native_batched_total_path_ms")
                cupy_knn_speedup = cupy_knn_speedups.get("cupy_knn_total")
                cases.append(
                    {
                        "batch_size": args.batch_size,
                        "mode": "fixed_batched",
                        **_stage8_case_metadata(args, "fixed_batched"),
                        "resolution_geometry_policy": _resolution_geometry_policy("fixed_batched"),
                        "n": args.n,
                        "queries": args.queries,
                        **_query_count_config(args),
                        "dim": dim,
                        "k": k,
                        "features": args.features,
                        "distribution": args.distribution,
                        "include_mls": include_mls,
                        "samples": samples,
                        "summary": summary,
                        "single_sample_loop_vs_native_batched_speedup": speedup,
                        "torch_cluster_baseline": torch_cluster_metadata,
                        "torch_cluster_total_vs_native_batched_speedup": torch_cluster_speedup,
                        "gpu_sorted_query_baseline": gpu_sorted_query_metadata,
                        "cupy_knn_baseline": cupy_knn_metadata,
                        "cupy_knn_total_vs_native_batched_speedup": cupy_knn_speedup,
                        "cupy_knn_speedups_vs_native_batched_total": cupy_knn_speedups,
                        "peak_gpu_memory_bytes": peak_memory,
                        "cuda_allocator_memory": _summarize_cuda_memory_samples(memory_samples),
                        "tensor_footprint_estimates": {
                            "query_outputs": _query_output_footprint(
                                batch_size=args.batch_size,
                                queries_per_sample=args.queries,
                                k=k,
                                dim=dim,
                                include_source_position_gather=True,
                            ),
                            "mls": _mls_tensor_footprint(
                                batch_size=args.batch_size,
                                queries_per_sample=args.queries,
                                k=k,
                                dim=dim,
                                channels=args.features,
                            )
                            if include_mls
                            else None,
                        },
                    }
                )
                continue

            points, query_points, features = _make_inputs(
                torch,
                args.n,
                args.queries,
                dim,
                args.features,
                args.seed + 1000 * dim + k,
                args.distribution,
            )

            for _ in range(args.warmup):
                bvh = torchbvh.build_bvh(points)
                try:
                    torchbvh.query_knn(bvh, query_points, k, source_points=points)
                finally:
                    torchbvh.destroy_bvh(bvh)
                if include_mls:
                    torchbvh.bvh_mls_interpolate(points, query_points, features, k=k)
                if torch_cluster_knn is not None:
                    _run_torch_cluster_case(torch, torch_cluster_knn, points, query_points, k)
                if args.gpu_sort_queries:
                    _run_gpu_sorted_query_case(torch, torchbvh, points, query_points, k)
                if cp is not None and LBVHIndex is not None and cupy_knn_variants is not None:
                    _run_cupy_knn_variants(torch, cp, LBVHIndex, points, query_points, k, cupy_knn_variants)
                torch.cuda.synchronize()

            samples: dict[str, list[float]] = {}
            memory_samples: list[dict[str, int]] = []
            peak_memory = 0
            torch_cluster_metadata = None
            cupy_knn_metadata = None
            gpu_sorted_query_metadata = None
            cupy_reference_distances = None
            if cp is not None and LBVHIndex is not None and cupy_knn_variants is not None:
                cupy_reference_distances = _implicit_reference_distances(torch, torchbvh, points, query_points, k)
            for _ in range(args.iters):
                timings, memory, memory_stats = _run_gpu_case(
                    torch,
                    torchbvh,
                    points,
                    query_points,
                    features,
                    k,
                    include_mls,
                )
                _merge_timings(
                    samples,
                    timings,
                )
                memory_samples.append(memory_stats)
                peak_memory = max(peak_memory, memory)
                if args.gpu_sort_queries:
                    timings, sorted_memory, sorted_memory_stats, gpu_sorted_query_metadata = _run_gpu_sorted_query_case(
                        torch,
                        torchbvh,
                        points,
                        query_points,
                        k,
                    )
                    _merge_timings(samples, timings)
                    memory_samples.append(sorted_memory_stats)
                    peak_memory = max(peak_memory, sorted_memory)
                if cKDTree is not None:
                    _merge_timings(samples, _run_cpu_case(torch, cKDTree, points, query_points, k))
                if torch_cluster_knn is not None:
                    timings, torch_cluster_metadata = _run_torch_cluster_case(
                        torch,
                        torch_cluster_knn,
                        points,
                        query_points,
                        k,
                    )
                    _merge_timings(samples, timings)
                if cp is not None and LBVHIndex is not None and cupy_knn_variants is not None:
                    timings, cupy_knn_metadata = _run_cupy_knn_variants(
                        torch,
                        cp,
                        LBVHIndex,
                        points,
                        query_points,
                        k,
                        cupy_knn_variants,
                        reference_squared_distances=cupy_reference_distances,
                    )
                    _merge_timings(samples, timings)

            summary = summarize_timings(samples)
            gpu_total = summary["gpu_total_bvh_path_ms"]["mean_ms"]
            cpu_total = summary["cpu_total_kdtree_path_ms"]["mean_ms"] if "cpu_total_kdtree_path_ms" in summary else None
            speedup = (cpu_total / gpu_total) if gpu_total and cpu_total else None
            torch_cluster_speedup = None
            if torch_cluster_metadata is not None:
                torch_cluster_total = summary["torch_cluster_total_ms"]["mean_ms"]
                torch_cluster_speedup = (
                    torch_cluster_total / gpu_total
                    if gpu_total and torch_cluster_total
                    else math.nan
                )
            cupy_knn_speedups = _cupy_knn_speedups(summary, "gpu_total_bvh_path_ms")
            cupy_knn_speedup = cupy_knn_speedups.get("cupy_knn_total")
            cases.append(
                {
                    "n": args.n,
                    "mode": "single",
                    **_stage8_case_metadata(args, "single"),
                    "resolution_geometry_policy": _resolution_geometry_policy("single"),
                    "queries": args.queries,
                    **_query_count_config(args),
                    "dim": dim,
                    "k": k,
                    "features": args.features,
                    "distribution": args.distribution,
                    "include_mls": include_mls,
                    "samples": samples,
                    "summary": summary,
                    "cpu_total_vs_gpu_total_speedup": speedup,
                    "torch_cluster_baseline": torch_cluster_metadata,
                    "torch_cluster_total_vs_gpu_total_speedup": torch_cluster_speedup,
                    "gpu_sorted_query_baseline": gpu_sorted_query_metadata,
                    "cupy_knn_baseline": cupy_knn_metadata,
                    "cupy_knn_total_vs_gpu_total_speedup": cupy_knn_speedup,
                    "cupy_knn_speedups_vs_gpu_total": cupy_knn_speedups,
                    "peak_gpu_memory_bytes": peak_memory,
                    "cuda_allocator_memory": _summarize_cuda_memory_samples(memory_samples),
                    "tensor_footprint_estimates": {
                        "query_outputs": _query_output_footprint(
                            batch_size=1,
                            queries_per_sample=args.queries,
                            k=k,
                            dim=dim,
                            include_source_position_gather=True,
                        ),
                        "mls": _mls_tensor_footprint(
                            batch_size=1,
                            queries_per_sample=args.queries,
                            k=k,
                            dim=dim,
                            channels=args.features,
                        )
                        if include_mls
                        else None,
                    },
                    "large_n_branch_used": False,
                }
            )

    return {
        "config": {
            "n": args.n,
            "batch_size": args.batch_size,
            "queries": args.queries,
            **_top_level_query_count_config(args),
            "dim": args.dim,
            "k": args.k,
            "iters": args.iters,
            "warmup": args.warmup,
            "features": args.features,
            "distribution": args.distribution,
            "ragged_n": args.ragged_n,
            "ragged_queries": args.ragged_queries,
            "displaced_query": args.displaced_query,
            "displaced_query_heads": args.displaced_query_heads,
            "displaced_query_channels": args.displaced_query_channels,
            "cpu_baseline": not args.no_cpu_baseline,
            "gpu_sort_queries": args.gpu_sort_queries,
            "include_mls": include_mls,
            "seed": args.seed,
            "case_name": args.case_name,
            "result_source": "benchmark",
            "comparison_label": args.comparison_label,
            "torch_cluster_baseline": args.torch_cluster_baseline,
            "cupy_knn_baseline": args.cupy_knn_baseline,
            "cupy_knn_options": _cupy_knn_options(args),
            "cupy_knn_sort_mode": args.cupy_knn_sort_mode,
            "cupy_knn_option_variants": [
                {"label": label, "options": options}
                for label, options in _cupy_knn_option_variants(args)
            ],
            "precision_policy": _precision_policy(),
        },
        "environment": _environment(),
        "cases": cases,
    }


def _run_metadata_command(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return completed.stdout.strip() or completed.stderr.strip() or None


def _nvcc_version() -> str | None:
    version = _run_metadata_command(["nvcc", "--version"])
    if version is not None:
        return version
    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_path:
        candidate = Path(cuda_path) / "bin" / "nvcc.exe"
        if candidate.exists():
            return _run_metadata_command([str(candidate), "--version"])
    return None


def _git_metadata() -> dict[str, Any]:
    commit = _run_metadata_command(["git", "rev-parse", "HEAD"])
    status = _run_metadata_command(["git", "status", "--short"])
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_short": status,
    }


def _environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "os": platform.system(),
        "benchmark_command": " ".join(sys.argv),
        "benchmark_script_version": BENCHMARK_SCRIPT_VERSION,
        "git": _git_metadata(),
        "nvcc": _nvcc_version(),
        "nvidia_driver": _run_metadata_command(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ]
        ),
        "compile_arch": "sm_89",
    }
    try:
        import torch

        props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
        env.update(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "gpu_total_memory_bytes": props.total_memory if props is not None else None,
                "gpu_compute_capability": (
                    f"{props.major}.{props.minor}" if props is not None else None
                ),
            }
        )
    except Exception as exc:  # pragma: no cover - defensive metadata only.
        env["torch_error"] = str(exc)
    return env


def _format_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:9.3f}"


def _format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}x"


def _cupy_speedup_text(case: dict[str, Any], key: str, label: str) -> str:
    speedups = case.get(key) or {}
    if not speedups:
        return ""
    parts = []
    if "cupy_knn_total" in speedups:
        parts.append(f"cupy_knn/{label}={_format_ratio(speedups['cupy_knn_total'])}")
    for mode in ("sorted", "unsorted"):
        speedup_key = f"cupy_knn_{mode}_total"
        if speedup_key in speedups:
            parts.append(f"cupy_knn_{mode}/{label}={_format_ratio(speedups[speedup_key])}")
    return " " + " ".join(parts) if parts else ""


def _query_count_summary(case: dict[str, Any]) -> str:
    requested = case.get("requested_queries_per_sample", case.get("requested_queries"))
    effective = case.get("effective_queries_per_sample", case.get("queries"))
    source = case.get("query_count_source", "queries")
    if requested == effective:
        return f"queries_per_sample={effective}"
    return f"requested_queries_per_sample={requested} effective_queries_per_sample={effective} query_source={source}"


def print_summary(result: dict[str, Any]) -> None:
    for case in result["cases"]:
        summary = case["summary"]
        if case.get("mode") == "ragged":
            print(
                f"ragged batch={case['batch_size']} total_n={case['total_n']} "
                f"total_queries={case['total_queries']} {_query_count_summary(case)} "
                f"dim={case['dim']} k={case['k']} "
                f"distribution={case['distribution']} "
                f"peak_mem={case['peak_gpu_memory_bytes'] / (1024 ** 2):.1f} MiB"
            )
            for key in (
                "ragged_build_ms",
                "ragged_query_traversal_ms",
                "ragged_position_gather_ms",
                "ragged_query_ms",
                "ragged_total_path_ms",
            ):
                if key in summary:
                    print(f"  {key:36s} mean={_format_ms(summary[key]['mean_ms'])} ms")
            continue

        if case.get("mode") == "displaced_query":
            torch_cluster_speedup = case.get("torch_cluster_total_vs_displaced_total_speedup")
            torch_cluster_text = (
                f" torch_cluster/package={torch_cluster_speedup:.2f}x"
                if torch_cluster_speedup is not None
                else ""
            )
            cupy_knn_text = _cupy_speedup_text(case, "cupy_knn_speedups_vs_displaced_total", "package")
            print(
                f"displaced_query batch={case['batch_size']} n={case['n']} "
                f"{_query_count_summary(case)} dim={case['dim']} k={case['k']} "
                f"heads={case['displaced_query_heads']} ch={case['displaced_query_channels']} "
                f"distribution={case['distribution']} "
                f"peak_mem={case['peak_gpu_memory_bytes'] / (1024 ** 2):.1f} MiB"
                f"{torch_cluster_text}"
                f"{cupy_knn_text}"
            )
            for key in (
                "displaced_query_batched_build_ms",
                "displaced_query_traversal_ms",
                "displaced_query_position_gather_ms",
                "displaced_query_ms",
                "displaced_query_value_gather_ms",
                "displaced_query_interpolate_ms",
                "displaced_query_total_path_ms",
                "gpu_sorted_bvh_build_ms",
                "gpu_sorted_query_sort_ms",
                "gpu_sorted_query_traversal_ms",
                "gpu_sorted_output_unsort_ms",
                "gpu_sorted_position_gather_ms",
                "gpu_sorted_bvh_query_ms",
                "gpu_sorted_total_bvh_path_ms",
                "torch_cluster_knn_ms",
                "torch_cluster_dense_conversion_ms",
                "torch_cluster_total_ms",
                "cupy_knn_build_ms",
                "cupy_knn_prepare_ms",
                "cupy_knn_query_ms",
                "cupy_knn_total_build_prepare_query_ms",
                "cupy_knn_sorted_cupy_knn_build_ms",
                "cupy_knn_sorted_cupy_knn_prepare_ms",
                "cupy_knn_sorted_cupy_knn_query_ms",
                "cupy_knn_sorted_cupy_knn_total_build_prepare_query_ms",
                "cupy_knn_unsorted_cupy_knn_build_ms",
                "cupy_knn_unsorted_cupy_knn_prepare_ms",
                "cupy_knn_unsorted_cupy_knn_query_ms",
                "cupy_knn_unsorted_cupy_knn_total_build_prepare_query_ms",
            ):
                if key in summary:
                    print(f"  {key:36s} mean={_format_ms(summary[key]['mean_ms'])} ms")
            continue

        if case.get("batch_size", 1) > 1:
            torch_cluster_speedup = case.get("torch_cluster_total_vs_native_batched_speedup")
            torch_cluster_text = (
                f" torch_cluster/native={torch_cluster_speedup:.2f}x"
                if torch_cluster_speedup is not None
                else ""
            )
            cupy_knn_text = _cupy_speedup_text(case, "cupy_knn_speedups_vs_native_batched_total", "native")
            print(
                f"batch={case['batch_size']} n={case['n']} {_query_count_summary(case)} "
                f"dim={case['dim']} k={case['k']} distribution={case['distribution']} "
                f"loop/native={case['single_sample_loop_vs_native_batched_speedup']:.2f}x "
                f"peak_mem={case['peak_gpu_memory_bytes'] / (1024 ** 2):.1f} MiB"
                f"{torch_cluster_text}"
                f"{cupy_knn_text}"
            )
            for key in (
                "native_batched_build_ms",
                "native_batched_query_traversal_ms",
                "native_batched_position_gather_ms",
                "native_batched_query_ms",
                "native_batched_mls_neighbor_feature_gather_ms",
                "native_batched_mls_solve_ms",
                "native_batched_mls_ms",
                "native_batched_total_path_ms",
                "native_batched_total_with_mls_ms",
                "single_sample_loop_total_path_ms",
                "single_sample_loop_total_with_mls_ms",
                "gpu_sorted_bvh_build_ms",
                "gpu_sorted_query_sort_ms",
                "gpu_sorted_query_traversal_ms",
                "gpu_sorted_output_unsort_ms",
                "gpu_sorted_position_gather_ms",
                "gpu_sorted_bvh_query_ms",
                "gpu_sorted_total_bvh_path_ms",
                "torch_cluster_knn_ms",
                "torch_cluster_dense_conversion_ms",
                "torch_cluster_total_ms",
                "cupy_knn_build_ms",
                "cupy_knn_prepare_ms",
                "cupy_knn_query_ms",
                "cupy_knn_total_build_prepare_query_ms",
                "cupy_knn_sorted_cupy_knn_build_ms",
                "cupy_knn_sorted_cupy_knn_prepare_ms",
                "cupy_knn_sorted_cupy_knn_query_ms",
                "cupy_knn_sorted_cupy_knn_total_build_prepare_query_ms",
                "cupy_knn_unsorted_cupy_knn_build_ms",
                "cupy_knn_unsorted_cupy_knn_prepare_ms",
                "cupy_knn_unsorted_cupy_knn_query_ms",
                "cupy_knn_unsorted_cupy_knn_total_build_prepare_query_ms",
            ):
                if key in summary:
                    print(f"  {key:36s} mean={_format_ms(summary[key]['mean_ms'])} ms")
            continue

        torch_cluster_speedup = case.get("torch_cluster_total_vs_gpu_total_speedup")
        torch_cluster_text = (
            f" torch_cluster/gpu={torch_cluster_speedup:.2f}x"
            if torch_cluster_speedup is not None
            else ""
        )
        cupy_knn_text = _cupy_speedup_text(case, "cupy_knn_speedups_vs_gpu_total", "gpu")
        cpu_speedup_text = _format_ratio(case.get("cpu_total_vs_gpu_total_speedup"))
        print(
            f"n={case['n']} {_query_count_summary(case)} dim={case['dim']} k={case['k']} "
            f"distribution={case['distribution']} "
            f"large_n_branch={case.get('large_n_branch_used', False)} "
            f"speedup={cpu_speedup_text} "
            f"peak_mem={case.get('peak_gpu_memory_bytes', 0) / (1024 ** 2):.1f} MiB"
            f"{torch_cluster_text}"
            f"{cupy_knn_text}"
        )
        for key in (
            "cpu_transfer_to_host_ms",
            "cpu_kdtree_build_ms",
            "cpu_kdtree_query_ms",
            "cpu_transfer_to_device_ms",
            "cpu_total_kdtree_path_ms",
            "gpu_bvh_build_ms",
            "gpu_bvh_query_traversal_ms",
            "gpu_position_gather_ms",
            "gpu_bvh_query_ms",
            "gpu_mls_neighbor_feature_gather_ms",
            "gpu_mls_solve_ms",
            "gpu_mls_interpolate_ms",
            "gpu_total_bvh_path_ms",
            "gpu_total_with_mls_ms",
            "gpu_sorted_bvh_build_ms",
            "gpu_sorted_query_sort_ms",
            "gpu_sorted_query_traversal_ms",
            "gpu_sorted_output_unsort_ms",
            "gpu_sorted_position_gather_ms",
            "gpu_sorted_bvh_query_ms",
            "gpu_sorted_total_bvh_path_ms",
            "torch_cluster_knn_ms",
            "torch_cluster_dense_conversion_ms",
            "torch_cluster_total_ms",
            "cupy_knn_build_ms",
            "cupy_knn_prepare_ms",
            "cupy_knn_query_ms",
            "cupy_knn_total_build_prepare_query_ms",
            "cupy_knn_sorted_cupy_knn_build_ms",
            "cupy_knn_sorted_cupy_knn_prepare_ms",
            "cupy_knn_sorted_cupy_knn_query_ms",
            "cupy_knn_sorted_cupy_knn_total_build_prepare_query_ms",
            "cupy_knn_unsorted_cupy_knn_build_ms",
            "cupy_knn_unsorted_cupy_knn_prepare_ms",
            "cupy_knn_unsorted_cupy_knn_query_ms",
            "cupy_knn_unsorted_cupy_knn_total_build_prepare_query_ms",
        ):
            if key in summary:
                print(f"  {key:28s} mean={_format_ms(summary[key]['mean_ms'])} ms")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_benchmark(args)
    print_summary(result)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
