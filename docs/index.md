# torchbvh

GPU-native geometry primitives for PyTorch point-cloud workflows. `torchbvh` provides
BVH construction, exact k-NN search, MLS interpolation, displaced-query helpers, and
FPS downsampling geometry as a model-agnostic CUDA extension — keeping spatial queries
on the GPU so they compose naturally with the rest of a PyTorch model.

## Performance

k-NN at N=10k, 3D, k=8 (RTX 3500 Ada, uniform distribution):

| | Build + query |
|---|---|
| `scipy_cKD-Tree` CPU | ~23 ms |
| `torch_cluster` GPU | ~6.8 ms |
| `cupy_knn` GPU | ~4.1 ms |
| `torchbvh` GPU | ~1.4 ms |

FPS at B=16, N=10k, 25% selection (RTX 3500 Ada):

| | Time |
|---|---|
| `fpsample` CPU | ~833 ms |
| `torch_fpsample` h=7 (CPU, fastest setting) | ~37 ms |
| `torchbvh` GPU | ~21 ms |

See `benchmarks/third_party_algorithm_comparison.ipynb` for optional comparisons against
`scipy`, `cupy-knn`, `torch_fpsample`, and `fpsample`.

## Install

```bash
pip install torchbvh
```

`torchbvh` builds a PyTorch CUDA extension. Source installs require PyTorch, a
compatible CUDA toolkit/NVCC, and a supported host compiler in the build environment.

## Quickstart

```python
import torch
import torchbvh as tb

points = torch.randn(1024, 3, device="cuda")
bvh = tb.BVH(points)

# k-NN
idx, dists = bvh.knn(points, k=8)         # (N,8) int64, (N,8) float32

# MLS interpolation — gradients flow through features
feat = torch.randn(1024, 16, device="cuda", requires_grad=True)
out = bvh.interpolate(points, feat, k=8)  # (N, 16)

# FPS downsampling geometry
fps = tb.fps(points, target_tokens=256)
# fps.indices, fps.points, fps.nearest_anchor, fps.anchor_radius, ...

# Batched: pass (B, N, D) → returns (B, N, k)
```

Supports `D in {2, 3}`, `k in {4, 8, 16}`, and CUDA float32 inputs. Public APIs
accept non-contiguous PyTorch views and normalize layout internally when needed.

## Navigation

- [User Guide](user_guide.md) — workflows for k-NN, MLS, displaced queries, and FPS
- [API Reference](api_reference.md) — signatures, shapes, and contracts for all public APIs
- [Lifecycle & Gradients](lifecycle_and_gradients.md) — handle ownership and autograd boundaries
- [Performance](performance.md) — benchmark interpretation and target workload
- [Examples](examples.md) — notebook-style examples and how to run them
