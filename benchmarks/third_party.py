"""Third-party production comparisons used by the companion notebook.

Inputs are staged before timing. Every timed call includes its own index build
or interpolation setup, but excludes data generation and device transfers.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from time import perf_counter

import cupy as cp
from cupyx.scipy.spatial import KDTree as CuPyKDTree
import fpsample
import numpy as np
import scipy
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F
import torch_cluster
import torch_fpsample

import torchbvh


POINT_COUNTS = (10_000, 20_000, 30_000, 40_000, 50_000)
LATTICE_SHAPES = {
    2: {
        10_000: (100, 100),
        20_000: (125, 160),
        30_000: (150, 200),
        40_000: (200, 200),
        50_000: (200, 250),
    },
    3: {
        10_000: (20, 20, 25),
        20_000: (20, 25, 40),
        30_000: (25, 30, 40),
        40_000: (25, 40, 40),
        50_000: (40, 25, 50),
    },
}
METHODS = {
    "knn": ("torchbvh", "scipy_cKDTree", "torch_cluster", "cupy_KDTree"),
    "fps": ("torchbvh_exact", "torchbvh_approx", "fpsample_h7", "torch_fpsample_h7"),
    "interpolation": ("torchbvh_mls", "grid_sample"),
}


@dataclass(frozen=True)
class Case:
    batch: int
    dim: int
    points: int

    @property
    def queries(self) -> int:
        return self.points

    @property
    def samples(self) -> int:
        return self.points // 4


def environment() -> dict:
    """Record enough metadata to interpret results from another machine."""
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "scipy": scipy.__version__,
        "cupy": cp.__version__,
        "torch_cluster": torch_cluster.__version__,
        "fpsample": metadata.version("fpsample"),
        "torch_fpsample": metadata.version("torch-fpsample"),
        "torchbvh": metadata.version("torchbvh"),
        "dtype": "float32",
        "knn_k": 4,
        "mls_k": 4,
        "channels": 64,
        "fps_ratio": 0.25,
        "timing": "median synchronized wall-clock ms; one warmup, three repeats by default",
        "transfers": "excluded; inputs prestaged in native CPU/GPU memory",
    }


def cases() -> list[Case]:
    return [Case(batch, dim, n) for batch in (1, 16) for dim in (2, 3) for n in POINT_COUNTS]


def _synchronize() -> None:
    # Device-wide synchronization also waits for CuPy kernels on this GPU.
    torch.cuda.synchronize()


def _time_call(fn, *, gpu: bool, warmup: int, repeats: int):
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative")
    for _ in range(warmup):
        result = fn()
        if gpu:
            _synchronize()
        del result
    samples = []
    for _ in range(repeats):
        if gpu:
            _synchronize()
        start = perf_counter()
        result = fn()
        if gpu:
            _synchronize()
        samples.append(1e3 * (perf_counter() - start))
    return float(np.median(samples)), result


def _row(case: Case, operation: str, method: str, **values) -> dict:
    return {
        "operation": operation,
        "method": method,
        "batch": case.batch,
        "dim": case.dim,
        "points": case.points,
        "queries": case.queries,
        "samples": case.samples if operation == "fps" else None,
        "channels": 64 if operation == "interpolation" else None,
        "status": "ok",
        "error": None,
        **values,
    }


def _failed(case: Case, operation: str, method: str, exc: Exception) -> dict:
    return _row(case, operation, method, status="failed", error=f"{type(exc).__name__}: {exc}")


def _point_inputs(case: Case) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, cp.ndarray, cp.ndarray]:
    rng = np.random.default_rng(20260923 + case.dim * 10 + case.points + case.batch)
    points_cpu = rng.random((case.batch, case.points, case.dim), dtype=np.float32)
    queries_cpu = rng.random((case.batch, case.queries, case.dim), dtype=np.float32)
    points_gpu = torch.from_numpy(points_cpu).to("cuda")
    queries_gpu = torch.from_numpy(queries_cpu).to("cuda")
    points_cupy = cp.asarray(points_cpu)
    queries_cupy = cp.asarray(queries_cpu)
    return points_cpu, queries_cpu, points_gpu, queries_gpu, points_cupy, queries_cupy


def _knn_reference(points: np.ndarray, queries: np.ndarray, limit: int = 256) -> np.ndarray:
    return np.stack([
        cKDTree(p).query(q[:limit], k=4, workers=1)[0]
        for p, q in zip(points, queries)
    ])


def _knn_distances(method: str, result, points_gpu: torch.Tensor, queries_gpu: torch.Tensor,
                   points_cpu: np.ndarray, queries_cpu: np.ndarray, limit: int = 256) -> np.ndarray:
    batch, count, _ = points_cpu.shape
    if method == "scipy_cKDTree":
        return np.stack([r[0][:limit] for r in result])
    if method == "cupy_KDTree":
        return np.stack([cp.asnumpy(r[0][:limit]) for r in result])
    if method == "torchbvh":
        return result[1][:, :limit].clamp_min(0).sqrt().cpu().numpy()
    edges = result
    if edges.shape != (2, batch * count * 4):
        raise AssertionError(f"torch_cluster returned shape {tuple(edges.shape)}")
    query_ids, source_ids = edges
    counts = torch.bincount(query_ids, minlength=batch * count)
    if not bool(torch.all(counts == 4)):
        raise AssertionError("torch_cluster did not return four neighbors per query")
    mask = query_ids.remainder(count) < limit
    selected_q = query_ids[mask]
    selected_s = source_ids[mask]
    ordered = torch.argsort(selected_q, stable=True)
    flat_points = points_gpu.reshape(batch * count, -1)
    flat_queries = queries_gpu.reshape(batch * count, -1)
    distances = (flat_points[selected_s] - flat_queries[selected_q]).square().sum(-1).sqrt()
    return distances[ordered].reshape(batch, limit, 4).cpu().numpy()


def run_knn(case: Case, *, warmup: int = 1, repeats: int = 3) -> list[dict]:
    p_cpu, q_cpu, p_gpu, q_gpu, p_cp, q_cp = _point_inputs(case)
    reference = _knn_reference(p_cpu, q_cpu)
    flat_p, flat_q = p_gpu.reshape(-1, case.dim), q_gpu.reshape(-1, case.dim)
    labels = torch.arange(case.batch, device="cuda").repeat_interleave(case.points)

    def bvh_call():
        with torchbvh.BVH(p_gpu[0] if case.batch == 1 else p_gpu) as bvh:
            idx, dist_sq = bvh.knn(q_gpu[0] if case.batch == 1 else q_gpu, k=4)
        if case.batch == 1:
            idx, dist_sq = idx.unsqueeze(0), dist_sq.unsqueeze(0)
        return idx, dist_sq

    def scipy_call():
        return [cKDTree(p).query(q, k=4, workers=1) for p, q in zip(p_cpu, q_cpu)]

    def cluster_call():
        if case.batch == 1:
            return torch_cluster.knn(flat_p, flat_q, k=4)
        return torch_cluster.knn(flat_p, flat_q, k=4, batch_x=labels, batch_y=labels,
                                 batch_size=case.batch)

    def cupy_call():
        return [CuPyKDTree(p).query(q, k=4) for p, q in zip(p_cp, q_cp)]

    calls = (bvh_call, scipy_call, cluster_call, cupy_call)
    rows = []
    for method, fn in zip(METHODS["knn"], calls):
        try:
            ms, result = _time_call(fn, gpu=method != "scipy_cKDTree", warmup=warmup, repeats=repeats)
            distances = np.sort(_knn_distances(method, result, p_gpu, q_gpu, p_cpu, q_cpu), axis=-1)
            if distances.shape != reference.shape or not np.isfinite(distances).all():
                raise AssertionError("invalid neighbor distance shape or values")
            error = np.abs(distances - reference)
            rows.append(_row(case, "knn", method, latency_ms=ms,
                             max_distance_error=float(error.max()),
                             neighbor_match_fraction=float(np.mean(np.all(error <= 1e-4, axis=-1)))))
            del result
        except Exception as exc:
            rows.append(_failed(case, "knn", method, exc))
    return rows


def _fps_indices(method: str, result, case: Case) -> np.ndarray:
    if method.startswith("torchbvh"):
        indices = result.indices.detach().cpu().numpy()
    elif method == "torch_fpsample_h7":
        indices = result[1].detach().cpu().numpy()
    else:
        indices = np.stack(result)
    if case.batch == 1 and indices.ndim == 1:
        indices = indices[None, :]
    if indices.shape != (case.batch, case.samples):
        raise AssertionError(f"sample indices have shape {indices.shape}")
    for row in indices:
        if len(np.unique(row)) != case.samples or np.any(row >= case.points) or np.any(row < 0):
            raise AssertionError("sample indices are duplicated or out of range")
    return indices


def run_fps(case: Case, *, warmup: int = 1, repeats: int = 3) -> list[dict]:
    p_cpu, _, p_gpu, _, _, _ = _point_inputs(case)
    p_native = p_gpu[0] if case.batch == 1 else p_gpu
    p_torch_cpu = torch.from_numpy(p_cpu)

    def bvh_exact():
        return torchbvh.fps(p_native, case.samples, mode="exact_bucketed", seed=0)

    def bvh_approx():
        return torchbvh.fps(p_native, case.samples, mode="approx_bucketed", seed=0)

    def fpsample_call():
        return [fpsample.bucket_fps_kdline_sampling(p, case.samples, h=7, start_idx=0)
                for p in p_cpu]

    def torch_fpsample_call():
        return torch_fpsample.sample(p_torch_cpu, case.samples, h=7, start_idx=0)

    rows = []
    for method, fn in zip(METHODS["fps"], (bvh_exact, bvh_approx, fpsample_call, torch_fpsample_call)):
        try:
            ms, result = _time_call(fn, gpu=method.startswith("torchbvh"), warmup=warmup, repeats=repeats)
            indices = _fps_indices(method, result, case)
            radii, means = [], []
            for p, idx in zip(p_cpu, indices):
                distances = cKDTree(p[idx]).query(p, k=1, workers=1)[0]
                radii.append(float(distances.max()))
                means.append(float(distances.mean()))
            rows.append(_row(case, "fps", method, latency_ms=ms,
                             seed_included_fraction=float(np.mean(np.any(indices == 0, axis=1))),
                             coverage_radius=max(radii), mean_coverage_distance=float(np.mean(means))))
            del result
        except Exception as exc:
            rows.append(_failed(case, "fps", method, exc))
    return rows


def _smooth_features(coords: torch.Tensor, channels: int = 64) -> torch.Tensor:
    channel = torch.arange(1, channels + 1, device=coords.device, dtype=coords.dtype)
    value = torch.zeros((*coords.shape[:-1], channels), device=coords.device, dtype=coords.dtype)
    for axis in range(coords.shape[-1]):
        frequency = (channel.remainder(5) + 1) * (0.3 + 0.1 * axis)
        value = value + torch.sin(coords[..., axis, None] * frequency) / coords.shape[-1]
    return value


def _lattice_inputs(case: Case):
    shape = LATTICE_SHAPES[case.dim][case.points]
    axes = [torch.linspace(-1, 1, steps, device="cuda", dtype=torch.float32) for steps in shape]
    mesh = torch.meshgrid(*axes, indexing="ij")
    # grid_sample expects (x, y) and (x, y, z), opposite tensor axis order.
    point_coords = torch.stack(tuple(reversed(mesh)), dim=-1).reshape(case.points, case.dim)
    points = point_coords.unsqueeze(0).expand(case.batch, -1, -1).contiguous()
    generator = torch.Generator(device="cuda").manual_seed(20260923 + case.batch + case.dim + case.points)
    queries = (torch.rand((case.batch, case.queries, case.dim), generator=generator,
                          device="cuda") * 1.9 - 0.95).contiguous()
    features = _smooth_features(points)
    grid_features = features.reshape(case.batch, *shape, 64).movedim(-1, 1).contiguous()
    grid_queries = queries.reshape(case.batch, case.queries, *((1,) * (case.dim - 1)), case.dim)
    truth = _smooth_features(queries)
    return points, queries, features, grid_features, grid_queries, truth


def run_interpolation(case: Case, *, warmup: int = 1, repeats: int = 3) -> list[dict]:
    points, queries, features, grid_features, grid_queries, truth = _lattice_inputs(case)

    def mls_call():
        return torchbvh.mls_interpolate(points, queries, features, k=4)

    def grid_call():
        values = F.grid_sample(grid_features, grid_queries, mode="bilinear",
                               padding_mode="border", align_corners=True)
        return values.reshape(case.batch, 64, case.queries).transpose(1, 2).contiguous()

    rows = []
    for method, fn in zip(METHODS["interpolation"], (mls_call, grid_call)):
        try:
            ms, result = _time_call(fn, gpu=True, warmup=warmup, repeats=repeats)
            if result.shape != truth.shape or not bool(torch.isfinite(result).all()):
                raise AssertionError("interpolation output has invalid shape or nonfinite values")
            delta = result - truth
            rows.append(_row(case, "interpolation", method, latency_ms=ms,
                             rmse=float(delta.square().mean().sqrt()),
                             max_abs_error=float(delta.abs().max())))
            del result
        except Exception as exc:
            rows.append(_failed(case, "interpolation", method, exc))
    return rows


def run_case(case: Case, *, warmup: int = 1, repeats: int = 3) -> list[dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch installation is required")
    rows = []
    for operation, runner in (("knn", run_knn), ("fps", run_fps),
                              ("interpolation", run_interpolation)):
        try:
            rows.extend(runner(case, warmup=warmup, repeats=repeats))
        except Exception as exc:
            rows.extend(_failed(case, operation, method, exc) for method in METHODS[operation])
    return rows
