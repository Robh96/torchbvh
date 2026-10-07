<h1 align="center"><img src="https://raw.githubusercontent.com/Robh96/torchbvh/main/docs/assets/torchbvh-logo.svg" alt="torchbvh" width="800"></h1>

**GPU geometry for PyTorch point clouds.** Fast k-NN, farthest-point sampling, interpolation, and ray tracing.

[![PyPI version](https://img.shields.io/pypi/v/torchbvh?label=PyPI&color=3855A5)](https://pypi.org/project/torchbvh/)
[![Python versions](https://img.shields.io/pypi/pyversions/torchbvh?color=3B7DBF)](https://pypi.org/project/torchbvh/)
[![Documentation](https://img.shields.io/readthedocs/torchbvh/latest?label=docs&color=17A398)](https://torchbvh.readthedocs.io/en/latest/)
[![License](https://img.shields.io/pypi/l/torchbvh?color=77C043)](https://github.com/Robh96/torchbvh/blob/main/LICENSE)

[Documentation](https://torchbvh.readthedocs.io/en/latest/) · [API reference](https://github.com/Robh96/torchbvh/blob/main/docs/api_reference.md) · [Benchmarks](https://github.com/Robh96/torchbvh/blob/main/examples/third_party_benchmarks.ipynb)

## Performance

**25.3x faster k-NN** than `torch_cluster` GPU and **9.2x faster approximate FPS** than `fpsample` CPU, averaged across all 20 benchmark workloads (arithmetic mean of per-workload ratios).

![torchbvh 0.3.2: batched 3D k-NN, FPS and regular-grid interpolation context.](https://raw.githubusercontent.com/Robh96/torchbvh/v0.3.2/docs/assets/performance_story_032.svg)

torchbvh k-NN/MLS rows were refreshed on 2026-10-07; the 160 competitor,
grid_sample and FPS rows retain their 2026-09-23 measurements. See the
[benchmark protocol and data](https://github.com/Robh96/torchbvh/blob/main/benchmarks/README.md)
and [MLS numerical behavior](https://github.com/Robh96/torchbvh/blob/main/docs/numerical_behavior.md).


## About

`torchbvh` provides CUDA operations for 2D and 3D float32 point clouds, including fixed-size batches:

- **k-NN:** BVH-accelerated nearest-neighbor search.
- **FPS:** exact and bucketed approximate farthest-point sampling.
- **Interpolation:** moving least squares (MLS) over scattered point features.
- **Ray tracing:** segment and triangle intersections.

## Installation

Install CUDA-enabled PyTorch, a matching CUDA toolkit with NVCC, and a supported C++ compiler, then install `torchbvh`:

```bash
pip install torchbvh
```

See the [build guide](https://github.com/Robh96/torchbvh/blob/main/docs/testing.md) for compiler and source-install details.

## Quickstart

```python
import torch
import torchbvh as tb

points = torch.rand(10_000, 3, device="cuda")
queries = torch.rand_like(points)

with tb.BVH(points) as bvh:
    neighbors, distances_sq = bvh.knn(queries, k=4)

sample = tb.fps(points, target_tokens=2_500, mode="approx_bucketed")
features = torch.rand(10_000, 64, device="cuda")
values = tb.mls_interpolate(points, queries, features, k=4)
```

See the [examples](https://github.com/Robh96/torchbvh/blob/main/docs/examples.md) for exact FPS, ray tracing, and gradients.

## Share point geometry within a forward

```python
other_queries = queries + 0.01
with tb.PointGeometry(points) as geometry:
    values = tb.mls_interpolate(points, queries, features, k=4, geometry=geometry)
    other = tb.mls_interpolate(points, other_queries, features, k=4, geometry=geometry)
# Saved tensors remain available for backward after the geometry closes.
```

Create a new geometry object each model forward. The original tensor must stay
unchanged while the object is open. Conditional MLS accepts `true_geometry`
and `false_geometry`; multihead MLS accepts `geometry`.
See [prepared geometry](https://github.com/Robh96/torchbvh/blob/main/docs/api_reference.md#pointgeometry)
for lifecycle rules and automatic dispatch eligibility.

## References

The BVH implementation builds on [Chitalu, Dubach, and Komura (2020)](https://doi.org/10.1111/cgf.13948) and [ImplicitBVH.jl](https://github.com/StellaOrg/ImplicitBVH.jl). Released under the [MIT license](https://github.com/Robh96/torchbvh/blob/main/LICENSE).
