<h1 align="center"><img src="https://raw.githubusercontent.com/Robh96/torchbvh/main/docs/assets/torchbvh-logo.svg" alt="torchbvh" width="800"></h1>

**GPU geometry for PyTorch point clouds.** Fast k-NN, farthest-point sampling, interpolation, and ray tracing.

[![PyPI version](https://img.shields.io/pypi/v/torchbvh?label=PyPI&color=3855A5)](https://pypi.org/project/torchbvh/)
[![Python versions](https://img.shields.io/pypi/pyversions/torchbvh?color=3B7DBF)](https://pypi.org/project/torchbvh/)
[![Documentation](https://img.shields.io/readthedocs/torchbvh/latest?label=docs&color=17A398)](https://torchbvh.readthedocs.io/en/latest/)
[![License](https://img.shields.io/pypi/l/torchbvh?color=77C043)](https://github.com/Robh96/torchbvh/blob/main/LICENSE)

[Documentation](https://torchbvh.readthedocs.io/en/latest/) · [API reference](https://github.com/Robh96/torchbvh/blob/main/docs/api_reference.md) · [Benchmarks](https://github.com/Robh96/torchbvh/blob/main/examples/third_party_benchmarks.ipynb)

## Performance

**20x faster k-NN** than `torch_cluster` GPU and **9.2x faster approximate FPS** than `fpsample` CPU, averaged across all 20 benchmark workloads.

![Batched 3D benchmark: torchbvh k-NN and FPS compared with third-party libraries; interpolation on a regular grid provides context.](https://raw.githubusercontent.com/Robh96/torchbvh/main/docs/assets/performance_story.svg)


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

## References

The BVH implementation builds on [Chitalu, Dubach, and Komura (2020)](https://doi.org/10.1111/cgf.13948) and [ImplicitBVH.jl](https://github.com/StellaOrg/ImplicitBVH.jl). Released under the [MIT license](https://github.com/Robh96/torchbvh/blob/main/LICENSE).
