"""Render the README performance graphic from a third-party benchmark CSV.

Run the notebook's full sweep, then export its ``results`` DataFrame with
``results.to_csv('artifacts/third_party_readme_2026-09-23.csv', index=False)``.

Example:
    python benchmarks/render_performance_story.py \
        artifacts/third_party_readme_2026-09-23.csv \
        docs/assets/performance_story
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


BG = "#091321"
CARD = "#112239"
STROKE = "#25425A"
TEXT = "#F4F9FF"
MUTED = "#ACBED1"
CYAN = "#54E2C2"
BLUE = "#92B3FF"
AMBER = "#FFBE74"
PINK = "#F494B4"


def _card(fig, x: float, y: float, width: float, height: float) -> None:
    fig.patches.append(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.002,rounding_size=0.022",
            transform=fig.transFigure,
            facecolor=CARD, edgecolor=STROKE, linewidth=1.2,
            zorder=-1,
        )
    )


def _trend(fig, frame: pd.DataFrame, *, x: float, y: float, width: float,
           height: float, xlabel: str,
           methods: list[tuple[str, str, str, str]]) -> None:
    ax = fig.add_axes((x, y, width, height), facecolor=CARD)
    for method, label, color, linestyle in methods:
        series = frame.loc[method].sort_index()
        ax.plot(series.index.to_numpy() / 1000, series.to_numpy(),
                color=color, linewidth=3.0 if method.startswith("torchbvh") else 2.4,
                linestyle=linestyle, marker="o", markersize=5.6,
                markeredgecolor=CARD, markeredgewidth=0.7,
                label=label, zorder=3 if method.startswith("torchbvh") else 2)
    ax.set_yscale("log")
    ax.set_xlim(8, 52)
    ax.set_xticks((10, 20, 30, 40, 50))
    ax.set_xlabel(xlabel, color=MUTED, fontsize=11, labelpad=8)
    ax.set_ylabel("MEDIAN MS  ·  LOG SCALE", color=MUTED, fontsize=11, labelpad=8)
    ax.grid(True, which="major", color="#345069", alpha=0.42, linewidth=0.8)
    ax.tick_params(axis="both", colors=MUTED, labelsize=11, length=0, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend_handles = [Line2D((0,), (0,), color=color, linewidth=3,
                             linestyle=linestyle, marker="o", markersize=5,
                             label=label)
                      for _, label, color, linestyle in methods]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.0, -0.34),
              ncol=2, frameon=False, fontsize=11.3, labelcolor=TEXT,
              handlelength=1.8, columnspacing=1.3, handletextpad=0.5,
              borderaxespad=0)


def render(results: Path, output_stem: Path) -> None:
    frame = pd.read_csv(results)
    subset = frame.loc[(frame["batch"] == 16) & (frame["dim"] == 3)]
    if (subset["status"] != "ok").any():
        raise ValueError("B=16, D=3 benchmark data contains failed methods")
    timings = subset.pivot(index="method", columns="points", values="latency_ms")
    required = {
        "torchbvh", "torch_cluster", "cupy_KDTree", "scipy_cKDTree",
        "torchbvh_approx", "torchbvh_exact", "fpsample_h7",
        "torch_fpsample_h7", "torchbvh_mls", "grid_sample",
    }
    if not required.issubset(timings.index) or set(timings.columns) != {
            10_000, 20_000, 30_000, 40_000, 50_000}:
        raise ValueError("Expected all methods at 10k-50k points for B=16, D=3")
    if not np.isfinite(timings.to_numpy()).all() or (timings.to_numpy() <= 0).any():
        raise ValueError("Benchmark latencies must be finite and positive")

    knn_gain = timings.loc["torch_cluster", 50_000] / timings.loc["torchbvh", 50_000]
    fps_gain = timings.loc["fpsample_h7", 50_000] / timings.loc["torchbvh_approx", 50_000]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "savefig.facecolor": BG,
        "svg.fonttype": "path",
    })
    fig = plt.figure(figsize=(16, 9.7), facecolor=BG)
    fig.text(0.05, 0.955, "TORCHBVH  /  PERFORMANCE", color=CYAN,
             fontsize=14, weight="bold", va="top")
    fig.text(0.05, 0.905, "Spatial search that scales.", color=TEXT,
             fontsize=38, weight="bold", va="top")
    fig.text(0.051, 0.835, "Production-size 3D point clouds · batch 16 · lower latency is better",
             color=MUTED, fontsize=15, va="top")

    _card(fig, 0.05, 0.282, 0.438, 0.515)
    _card(fig, 0.512, 0.282, 0.438, 0.515)
    fig.text(0.074, 0.765, "01  /  EXACT K-NEAREST NEIGHBORS", color=MUTED,
             fontsize=12, weight="bold", va="top")
    fig.text(0.074, 0.725, f"{knn_gain:.0f}×", color=CYAN,
             fontsize=47, weight="bold", va="top")
    fig.text(0.22, 0.698, "faster vs torch_cluster GPU\nat 50k points", color=TEXT,
             fontsize=13, linespacing=1.5, va="top")

    fig.text(0.536, 0.765, "02  /  FARTHEST-POINT SAMPLING", color=MUTED,
             fontsize=12, weight="bold", va="top")
    fig.text(0.536, 0.725, f"{fps_gain:.0f}×", color=CYAN,
             fontsize=47, weight="bold", va="top")
    fig.text(0.685, 0.698, "faster vs fpsample CPU\nat 50k points · 25% kept", color=TEXT,
             fontsize=13, linespacing=1.5, va="top")

    _trend(fig, timings, x=0.096, y=0.412, width=0.352, height=0.202,
           xlabel="POINTS / QUERIES (THOUSANDS)",
           methods=[
               ("torchbvh", "torchbvh", CYAN, "-"),
               ("torch_cluster", "torch_cluster", BLUE, "-"),
               ("cupy_KDTree", "CuPy KDTree", AMBER, "-"),
               ("scipy_cKDTree", "SciPy cKDTree", PINK, "-"),
           ])
    _trend(fig, timings, x=0.558, y=0.412, width=0.352, height=0.202,
           xlabel="POINTS (THOUSANDS)",
           methods=[
               ("torchbvh_approx", "torchbvh approx", CYAN, "-"),
               ("torchbvh_exact", "torchbvh exact", BLUE, "--"),
               ("torch_fpsample_h7", "torch_fpsample CPU", AMBER, "-"),
               ("fpsample_h7", "fpsample CPU", PINK, "-"),
           ])

    _card(fig, 0.05, 0.105, 0.9, 0.145)
    fig.text(0.075, 0.218, "03  /  INTERPOLATION CONTEXT", color=MUTED,
             fontsize=12, weight="bold", va="top")
    fig.text(0.075, 0.177, "MLS works directly on scattered points.", color=TEXT,
             fontsize=21, weight="bold", va="top")
    fig.text(0.075, 0.135,
             "On regular grids, grid_sample is faster.",
             color=MUTED, fontsize=12.7, va="top")
    fig.text(0.625, 0.225, "REGULAR-GRID REFERENCE  /  MEDIAN MS", color=MUTED,
             fontsize=10.5, weight="bold", va="top")
    context_ax = fig.add_axes((0.64, 0.128, 0.27, 0.078), facecolor=CARD)
    for method, label, color in (("grid_sample", "grid_sample", AMBER),
                                 ("torchbvh_mls", "MLS", BLUE)):
        series = timings.loc[method].sort_index()
        context_ax.plot(series.index.to_numpy() / 1000, series.to_numpy(),
                        color=color, linewidth=2.5, marker="o", markersize=4.5)
        context_ax.text(52, float(series.loc[50_000]),
                        f"{label}  {series.loc[50_000]:.0f} ms",
                        color=color, fontsize=10.5, va="center")
    context_ax.set_xlim(8, 78)
    context_ax.set_yscale("log")
    context_ax.set_ylim(2, 60)
    context_ax.set_xticks((10, 30, 50))
    context_ax.set_yticks(())
    context_ax.tick_params(axis="x", colors=MUTED, labelsize=9.5, length=0, pad=3)
    context_ax.grid(True, axis="x", color="#345069", alpha=0.45, linewidth=0.8)
    for spine in context_ax.spines.values():
        spine.set_visible(False)

    fig.text(0.051, 0.061,
             "RTX 3500 Ada  ·  float32  ·  warmed one-shot calls  ·  median of 3  ·  transfers excluded",
             color=MUTED, fontsize=11.5, va="center")
    fig.text(0.949, 0.061, "torchbvh", color=CYAN, fontsize=14,
             weight="bold", ha="right", va="center")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches=None)
    fig.savefig(output_stem.with_suffix(".png"), dpi=180, bbox_inches=None)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="CSV produced from the benchmark notebook")
    parser.add_argument("output_stem", type=Path, help="Output path without SVG/PNG extension")
    args = parser.parse_args()
    render(args.results, args.output_stem)


if __name__ == "__main__":
    main()
