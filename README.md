# torchbvh

GPU-native geometry primitives for PyTorch point-cloud workflows: BVH construction,
exact k-NN search, MLS interpolation, displaced-query helpers, and FPS downsampling.
Model-agnostic — no learned weights, no training loop, no architecture.

## Performance

k-NN at N=10k, 3D, k=8 (RTX 3500 Ada, uniform distribution):

| | Build + query |
|---|---|
| CPU k-d tree + transfers | ~23 ms |
| `torch_cluster` | ~12 ms |
| `torchbvh` | ~1.4 ms |

FPS at B=16, N=10k, 25% selection (RTX 3500 Ada):

| | Time |
|---|---|
| `torch_fpsample` h=7 (CPU, fastest setting) | ~35 ms |
| `torchbvh` | ~25 ms |

See `benchmarks/third_party_algorithm_comparison.ipynb` for optional comparisons against
`scipy`, `cupy-knn`, `torch_fpsample`, and `fpsample`.


## Install

```bash
pip install -e . --no-build-isolation
```

## Docs

Documentation is intended to be hosted at
[torchbvh.readthedocs.io](https://torchbvh.readthedocs.io/). The link is inactive until
the repository is public and the Read the Docs project has been activated.

```bash
pip install -r requirements-docs.txt && python -m mkdocs serve
```


## Quickstart

```python
import torch
import torchbvh as tb

points = torch.randn(1024, 3, device="cuda")
bvh = tb.BVH(points)

# k-NN
idx, dists = bvh.knn(points, k=8)         # (N,8) int32, (N,8) float32

# MLS interpolation — gradients flow through features
feat = torch.randn(1024, 16, device="cuda", requires_grad=True)
out = bvh.interpolate(points, feat, k=8)  # (N, 16)

# FPS downsampling geometry
fps = tb.fps(points, target_tokens=256)
# fps.indices, fps.points, fps.nearest_anchor, fps.anchor_radius, ...

# Batched: pass (B, N, D) → returns (B, N, k)
```

Supports `D in {2, 3}`, `k in {4, 8, 16}`, CUDA float32 contiguous inputs.
