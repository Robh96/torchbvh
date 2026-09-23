# torchbvh

GPU-native geometry primitives for PyTorch point-cloud workflows. `torchbvh` provides BVH construction, exact k-NN search, closest-hit ray tracing, MLS interpolation, displaced-query helpers, and FPS downsampling geometry as a model-agnostic CUDA extension.

## Performance

The production routes use cooperative BVH construction, cached-bound k-NN, indexed packed MLS, bucketed FPS, and specialized segment/general triangle ray traversal. See [Performance](performance.md) for maintained benchmark commands and reporting guidance.

## Install

Install CUDA-enabled PyTorch for your GPU, the matching CUDA toolkit (including NVCC), and a supported host C++ compiler. Then compile `torchbvh` against the PyTorch already installed in your environment:

```bash
python -m pip install --upgrade setuptools wheel
python -m pip install --no-build-isolation --no-binary torchbvh torchbvh
```

`torchbvh` is CUDA-only. Version 0.3.0 is distributed as source rather than as a GPU-specific wheel. PyTorch targets the GPUs visible while building by default; set `TORCH_CUDA_ARCH_LIST` first to target a different or broader set.
See [Testing](testing.md) for build and verification guidance.

## Quickstart

```python
import torch
import torchbvh as tb

points = torch.randn(128, 3, device="cuda")
features = torch.randn(128, 8, device="cuda", requires_grad=True)

with tb.BVH(points) as bvh:
    indices, dist_sq = bvh.knn(points[:16], k=8)  # (16, 8) each
    values = bvh.interpolate(points[:16], features, k=8)

values.sum().backward()  # gradients flow to features

samples = tb.fps(points, target_tokens=32)
samples.points           # (32, 3)
samples.nearest_anchor   # (128,)

# The same BVH class accepts fixed-size batches.
batched_points = torch.randn(2, 32, 3, device="cuda")
with tb.BVH(batched_points) as bvh:
    batch_indices, batch_dist_sq = bvh.knn(batched_points[:, :8], k=4)
# Both tensors have shape (2, 8, 4).
```

Supports `D in {2, 3}`, `k in {4, 8, 16}`, and CUDA float32 inputs.
Public APIs accept non-contiguous PyTorch views and normalize layout internally when needed.

## Navigation

- [User Guide](user_guide.md) — workflows for k-NN, MLS, displaced queries, and FPS
- [API Reference](api_reference.md) — signatures, shapes, and contracts for all public APIs
- [Lifecycle & Gradients](lifecycle_and_gradients.md) — handle ownership and autograd boundaries
- [Performance](performance.md) — benchmark interpretation and target workload
- [Testing & Release](testing.md) — CUDA tests and distribution checks
- [Examples](examples.md) — notebook-style examples and how to run them
