"""Stage 12 M8 Flower-style FPS-warp downsample benchmark.

This benchmark is intentionally standalone. It measures the public FPS geometry +
learned-warp interpolation path.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torchbvh  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    B: int
    N: int
    D: int
    H: int
    C: int
    M: int
    R: int


@dataclass
class RunResult:
    label: str
    case: str
    status: str
    error: str | None
    timings_ms: dict[str, float | None]
    output_shape: list[int] | None
    coarse_count: int | None
    output_finite: bool | None
    gradient_finite: bool | None
    coverage_radius: float | None
    mean_nearest_anchor_distance: float | None
    peak_memory_mib: float | None


@dataclass(frozen=True)
class SyntheticParams:
    amp_weight: torch.Tensor
    slot_dirs: torch.Tensor
    concat_weight: torch.Tensor
    concat_seed_weight: torch.Tensor
    attention_q_weight: torch.Tensor
    attention_k_weight: torch.Tensor
    attention_v_weight: torch.Tensor
    attention_out_weight: torch.Tensor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("small", "10648", "50000", "all"), default="small")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=15120)
    parser.add_argument("--k", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--mixer", choices=("concat", "attention", "both"), default="both")
    parser.add_argument("--skip-backward", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.iters < 1:
        raise SystemExit("--iters must be >= 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    return args


def make_case(name: str) -> Case:
    if name == "small":
        B, N, D, H, C = 2, 1024, 3, 4, 32
    elif name == "10648":
        B, N, D, H, C = 16, 10648, 3, 8, 160
    elif name == "50000":
        B, N, D, H, C = 16, 50000, 3, 8, 160
    else:
        raise ValueError(f"unknown case {name!r}")
    M = N // 4
    R = (H * N) // M
    return Case(name=name, B=B, N=N, D=D, H=H, C=C, M=M, R=R)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _cuda_event_ms(fn: Callable[[], torch.Tensor | object]) -> tuple[float, torch.Tensor | object]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out


def _reset_memory() -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _peak_mib() -> float:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024.0**2)


def _make_inputs(case: Case, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    points = torch.rand((case.B, case.N, case.D), device="cuda", dtype=torch.float32).contiguous()
    features = torch.randn((case.B, case.N, case.C), device="cuda", dtype=torch.float32).contiguous()
    features.requires_grad_(True)
    return points, features


def _fixed_weight(rows: int, cols: int, *, device: torch.device, seed: int, scale: float) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return (torch.randn((rows, cols), generator=gen, device=device, dtype=torch.float32) * scale).contiguous()


def _slot_directions(R: int, D: int, *, device: torch.device) -> torch.Tensor:
    slots = torch.arange(R, device=device, dtype=torch.float32)
    dirs = torch.stack(
        (
            torch.sin(slots * 1.61803398875),
            torch.cos(slots * 2.41421356237),
            torch.sin(slots * 0.70710678118 + 0.3),
        ),
        dim=-1,
    )[:, :D]
    return torch.nn.functional.normalize(dirs, dim=-1).contiguous()


def _make_synthetic_params(case: Case, *, device: torch.device, seed: int) -> SyntheticParams:
    hidden = min(64, case.C)
    return SyntheticParams(
        amp_weight=_fixed_weight(case.C, case.R, device=device, seed=seed + 17, scale=1.0 / max(case.C, 1) ** 0.5),
        slot_dirs=_slot_directions(case.R, case.D, device=device),
        concat_weight=_fixed_weight(
            case.R * case.C,
            case.C,
            device=device,
            seed=seed + 29,
            scale=1.0 / max(case.R * case.C, 1) ** 0.5,
        ),
        concat_seed_weight=_fixed_weight(
            case.C,
            case.C,
            device=device,
            seed=seed + 31,
            scale=1.0 / max(case.C, 1) ** 0.5,
        ),
        attention_q_weight=_fixed_weight(
            case.C,
            hidden,
            device=device,
            seed=seed + 41,
            scale=1.0 / max(case.C, 1) ** 0.5,
        ),
        attention_k_weight=_fixed_weight(
            case.C,
            hidden,
            device=device,
            seed=seed + 43,
            scale=1.0 / max(case.C, 1) ** 0.5,
        ),
        attention_v_weight=_fixed_weight(
            case.C,
            case.C,
            device=device,
            seed=seed + 47,
            scale=1.0 / max(case.C, 1) ** 0.5,
        ),
        attention_out_weight=_fixed_weight(
            case.C,
            case.C,
            device=device,
            seed=seed + 53,
            scale=1.0 / max(case.C, 1) ** 0.5,
        ),
    )


def _build_offsets(
    seed_features: torch.Tensor,
    anchor_radius: torch.Tensor,
    R: int,
    D: int,
    params: SyntheticParams,
) -> torch.Tensor:
    B, M, C = seed_features.shape
    amplitudes = torch.tanh(seed_features.reshape(B * M, C).matmul(params.amp_weight)).reshape(B, M, R)
    scale = anchor_radius.clamp_min(1.0e-12).sqrt().clamp_min(1.0e-4)
    return (0.25 * scale[:, :, None, None] * amplitudes[:, :, :, None] * params.slot_dirs[None, None]).contiguous()


def _concat_mixer(sampled: torch.Tensor, seed_features: torch.Tensor, params: SyntheticParams) -> torch.Tensor:
    B, M, R, C = sampled.shape
    flat = sampled.reshape(B * M, R * C)
    mixed = flat.matmul(params.concat_weight).reshape(B, M, C)
    mixed = mixed + seed_features.reshape(B * M, C).matmul(params.concat_seed_weight).reshape(B, M, C)
    return torch.nn.functional.gelu(mixed).contiguous()


def _attention_mixer(
    sampled: torch.Tensor,
    seed_features: torch.Tensor,
    offsets: torch.Tensor,
    anchor_radius: torch.Tensor,
    params: SyntheticParams,
) -> torch.Tensor:
    B, M, R, C = sampled.shape
    hidden = params.attention_q_weight.size(1)

    offset_scale = anchor_radius.clamp_min(1.0e-12).sqrt().clamp_min(1.0e-4)[:, :, None, None]
    offset_bias = (offsets / offset_scale).square().sum(dim=-1, keepdim=True)
    query = seed_features.reshape(B * M, C).matmul(params.attention_q_weight).reshape(B, M, 1, hidden)
    keys = sampled.reshape(B * M * R, C).matmul(params.attention_k_weight).reshape(B, M, R, hidden)
    values = sampled.reshape(B * M * R, C).matmul(params.attention_v_weight).reshape(B, M, R, C)
    scores = (query * keys).sum(dim=-1) / hidden**0.5 - 0.05 * offset_bias.squeeze(-1)
    weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
    attended = (weights * values).sum(dim=2)
    return torch.nn.functional.gelu(
        attended.reshape(B * M, C).matmul(params.attention_out_weight).reshape(B, M, C)
    ).contiguous()


def _run_fps_warp_once(
    case: Case,
    points: torch.Tensor,
    features: torch.Tensor,
    *,
    k: int,
    mixer: str,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float], dict[str, object]]:
    timings: dict[str, float] = {}
    metadata: dict[str, object] = {}
    params = _make_synthetic_params(case, device=features.device, seed=seed)

    timings["fps_geometry_ms"], fps = _cuda_event_ms(
        lambda: torchbvh.fps(points, case.M, seed=0)
    )
    metadata["coverage_radius"] = float(fps.nearest_anchor_dist_sq.max().clamp_min(0).sqrt().item())
    metadata["mean_nearest_anchor_distance"] = float(fps.nearest_anchor_dist_sq.clamp_min(0).sqrt().mean().item())

    def gather_seed() -> torch.Tensor:
        return torch.gather(features, 1, fps.indices[:, :, None].expand(case.B, case.M, case.C)).contiguous()

    timings["fps_seed_gather_ms"], seed_features = _cuda_event_ms(gather_seed)
    timings["offset_construction_ms"], offsets = _cuda_event_ms(
        lambda: _build_offsets(seed_features, fps.anchor_radius, case.R, case.D, params)
    )
    timings["query_construction_ms"], queries = _cuda_event_ms(
        lambda: (fps.points[:, :, None, :] + offsets).reshape(case.B, case.M * case.R, case.D).contiguous()
    )
    timings["interpolation_ms"], sampled_flat = _cuda_event_ms(
        lambda: torchbvh.bvh_mls_interpolate_batched(points, queries, features, k=k)
    )
    sampled = sampled_flat.reshape(case.B, case.M, case.R, case.C)

    if mixer == "concat":
        timings["mixer_ms"], output = _cuda_event_ms(lambda: _concat_mixer(sampled, seed_features, params))
    elif mixer == "attention":
        timings["mixer_ms"], output = _cuda_event_ms(
            lambda: _attention_mixer(sampled, seed_features, offsets, fps.anchor_radius, params)
        )
    else:
        raise ValueError(f"unknown mixer {mixer!r}")
    timings["total_forward_ms"] = sum(timings.values())
    return output, timings, metadata


def _run_measured(
    label: str,
    case: Case,
    seed: int,
    warmup: int,
    iters: int,
    skip_backward: bool,
    fn: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, dict[str, float], dict[str, object]]],
) -> RunResult:
    samples: dict[str, list[float]] = {}
    last_output = None
    last_metadata: dict[str, object] = {}
    backward_samples: list[float] = []
    gradient_finite: bool | None = None

    try:
        _reset_memory()
        for iteration in range(warmup + iters):
            points, features = _make_inputs(case, seed + iteration)
            output, timings, metadata = fn(points, features)
            if not skip_backward:
                backward_ms, _ = _cuda_event_ms(lambda: output.square().mean().backward())
                if iteration >= warmup:
                    backward_samples.append(backward_ms)
                gradient_finite = bool(features.grad is not None and torch.isfinite(features.grad).all().item())
            if iteration >= warmup:
                for key, value in timings.items():
                    samples.setdefault(key, []).append(float(value))
            last_output = output
            last_metadata = metadata
            del points, features, output
            torch.cuda.synchronize()

        assert last_output is not None
        timing_summary = {key: _mean(values) for key, values in samples.items()}
        if backward_samples:
            timing_summary["backward_ms"] = _mean(backward_samples)
            timing_summary["train_step_ms"] = (
                (timing_summary.get("total_forward_ms") or 0.0) + (timing_summary.get("backward_ms") or 0.0)
            )
        return RunResult(
            label=label,
            case=case.name,
            status="ok",
            error=None,
            timings_ms=timing_summary,
            output_shape=list(last_output.shape),
            coarse_count=int(last_output.size(1)),
            output_finite=bool(torch.isfinite(last_output).all().item()),
            gradient_finite=gradient_finite if not skip_backward else None,
            coverage_radius=last_metadata.get("coverage_radius"),  # type: ignore[arg-type]
            mean_nearest_anchor_distance=last_metadata.get("mean_nearest_anchor_distance"),  # type: ignore[arg-type]
            peak_memory_mib=_peak_mib(),
        )
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.synchronize()
        peak = _peak_mib()
        torch.cuda.empty_cache()
        gc.collect()
        return RunResult(
            label=label,
            case=case.name,
            status="oom",
            error=str(exc).splitlines()[0],
            timings_ms={},
            output_shape=None,
            coarse_count=None,
            output_finite=None,
            gradient_finite=None,
            coverage_radius=None,
            mean_nearest_anchor_distance=None,
            peak_memory_mib=peak,
        )
    except Exception as exc:
        torch.cuda.empty_cache()
        gc.collect()
        return RunResult(
            label=label,
            case=case.name,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            timings_ms={},
            output_shape=None,
            coarse_count=None,
            output_finite=None,
            gradient_finite=None,
            coverage_radius=None,
            mean_nearest_anchor_distance=None,
            peak_memory_mib=None,
        )


def run_case(args: argparse.Namespace, case: Case) -> list[RunResult]:
    mixers = ["concat", "attention"] if args.mixer == "both" else [args.mixer]
    results: list[RunResult] = []
    for mixer in mixers:
        results.append(
            _run_measured(
                f"fps_warp_public_{mixer}",
                case,
                args.seed,
                args.warmup,
                args.iters,
                args.skip_backward,
                lambda points, features, mixer=mixer: _run_fps_warp_once(
                    case,
                    points,
                    features,
                    k=args.k,
                    mixer=mixer,
                    seed=args.seed,
                ),
            )
        )
    return results


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def print_results(case: Case, results: list[RunResult]) -> None:
    print(
        f"\nStage 12 M8 case={case.name} B={case.B} N={case.N} D={case.D} "
        f"H={case.H} C={case.C} M={case.M} R={case.R}"
    )
    print(
        "| Path | Status | FPS geom | Gather | Offset | Query | Interp | Mixer | "
        "Forward | Backward | Peak MiB | Output | Grad finite | Shape | Error |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|")
    for result in results:
        t = result.timings_ms
        shape = "n/a" if result.output_shape is None else "x".join(str(v) for v in result.output_shape)
        print(
            f"| {result.label} | {result.status} | {_fmt(t.get('fps_geometry_ms'))} | "
            f"{_fmt(t.get('fps_seed_gather_ms'))} | {_fmt(t.get('offset_construction_ms'))} | "
            f"{_fmt(t.get('query_construction_ms'))} | "
            f"{_fmt(t.get('interpolation_ms'))} | {_fmt(t.get('mixer_ms'))} | "
            f"{_fmt(t.get('total_forward_ms'))} | {_fmt(t.get('backward_ms'))} | "
            f"{_fmt(result.peak_memory_mib)} | {result.output_finite} | "
            f"{result.gradient_finite} | {shape} | {result.error or ''} |"
        )
    for result in results:
        if result.coverage_radius is not None:
            print(
                f"  {result.label}: coverage_radius={result.coverage_radius:.6f} "
                f"mean_nearest_anchor_distance={result.mean_nearest_anchor_distance:.6f}"
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for benchmark_fps.py")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    names = ["small", "10648", "50000"] if args.case == "all" else [args.case]
    payload = {"config": vars(args), "cases": []}
    for name in names:
        case = make_case(name)
        results = run_case(args, case)
        print_results(case, results)
        payload["cases"].append({"case": asdict(case), "results": [asdict(result) for result in results]})

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
