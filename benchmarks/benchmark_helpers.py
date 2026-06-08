from __future__ import annotations

import gc
import importlib
import math
import platform
import statistics
import sys
import time
from typing import Callable

import torch


class SkipBackend(RuntimeError):
    pass


_MODULE_CACHE = {}


def optional_module(name: str):
    try:
        return _MODULE_CACHE[name]
    except KeyError:
        pass
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        raise SkipBackend(f"{name} is not available: {exc}")
    _MODULE_CACHE[name] = module
    return module


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def dense_cdist_skip(batch: int, queries: int, points: int, max_elements: int) -> str | None:
    elements = batch * queries * points
    if elements <= max_elements:
        return None
    gib = elements * 4 / (1024**3)
    return f"skipped: dense cdist matrix would be {gib:.1f} GiB"


def make_random_points_queries(
    *,
    device: str,
    seed: int,
    batch_size: int,
    n_points: int,
    n_queries: int,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    points = torch.rand(batch_size, n_points, dim, device=device, generator=generator)
    queries = torch.rand(batch_size, n_queries, dim, device=device, generator=generator)
    return points.contiguous(), queries.contiguous()


def environment(torchbvh_module, extension_module, device: str) -> dict:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torchbvh": torchbvh_module.__file__,
        "torchbvh_extension": extension_module.__file__,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def patch_cupy_knn_compile_flags(cp, lbvh_module) -> None:
    flags = tuple(getattr(lbvh_module, "_compile_flags", ()))
    old_standards = {"--std=c++11", "--std=c++14", "-std=c++11", "-std=c++14"}
    patched = tuple("--std=c++17" if flag in old_standards else flag for flag in flags)
    if patched == flags:
        return

    lbvh_module._compile_flags = patched
    if hasattr(lbvh_module, "_lbvh_src"):
        lbvh_module._construct_tree_kernels = cp.RawModule(
            code=lbvh_module._lbvh_src,
            options=patched,
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


def time_once(fn: Callable):
    start_event = end_event = None
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    sync()
    start = time.perf_counter()
    if start_event is not None:
        start_event.record()
    result = fn()
    if end_event is not None:
        end_event.record()
    sync()

    wall_ms = (time.perf_counter() - start) * 1000.0
    event_ms = float(start_event.elapsed_time(end_event)) if start_event is not None else None
    return result, wall_ms, event_ms


def summarize(values: list[float], prefix: str = "") -> dict:
    return {
        f"{prefix}mean_ms": sum(values) / len(values),
        f"{prefix}min_ms": min(values),
        f"{prefix}max_ms": max(values),
        f"{prefix}median_ms": statistics.median(values),
    }


def skipped_row(label: str, group: str, reason: str, *, warmup: int, **extra) -> dict:
    return {
        "group": group,
        "backend": label,
        "status": "skipped",
        "reason": reason,
        "warmup_runs": warmup,
        "timed_runs": 0,
        "mean_ms": math.nan,
        "min_ms": math.nan,
        "max_ms": math.nan,
        "median_ms": math.nan,
        "peak_cuda_mib": math.nan,
        **extra,
    }


def benchmark(
    label: str,
    group: str,
    fn: Callable,
    quality_fn: Callable,
    *,
    warmup: int,
    iters: int,
) -> dict:
    try:
        for _ in range(warmup):
            fn()
        sync()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        result = None
        wall_ms = []
        event_ms = []
        for _ in range(iters):
            result, wall, event = time_once(fn)
            wall_ms.append(wall)
            if event is not None:
                event_ms.append(event)

        row = {
            "group": group,
            "backend": label,
            "status": "ok",
            "warmup_runs": warmup,
            "timed_runs": iters,
            **summarize(wall_ms),
            "timings_ms": wall_ms,
            "cuda_event_timings_ms": event_ms or None,
            "peak_cuda_mib": torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None,
        }
        if event_ms:
            row.update(summarize(event_ms, "cuda_event_"))
        row.update(quality_fn(result))
        return row
    except torch.cuda.OutOfMemoryError as exc:
        gc.collect()
        torch.cuda.empty_cache()
        return {"group": group, "backend": label, "status": "oom", "reason": str(exc).splitlines()[0]}
    except SkipBackend as exc:
        return skipped_row(label, group, str(exc), warmup=warmup)
    except Exception as exc:
        return {"group": group, "backend": label, "status": "error", "reason": repr(exc)}


def run_backends(
    backends: dict,
    group: str,
    quality_fn: Callable,
    *,
    warmup: int,
    iters: int,
    skip_fn: Callable[[str], str | None] | None = None,
) -> list[dict]:
    rows = []
    for label, fn in backends.items():
        reason = skip_fn(label) if skip_fn is not None else None
        if reason is None:
            rows.append(benchmark(label, group, fn, quality_fn, warmup=warmup, iters=iters))
        else:
            rows.append(skipped_row(label, group, reason, warmup=warmup))
    return rows


def show_rows(rows: list[dict]) -> list[dict]:
    try:
        import pandas as pd
        from IPython.display import display

        display(pd.DataFrame(rows))
    except Exception:
        for row in rows:
            print(row)
    return rows
