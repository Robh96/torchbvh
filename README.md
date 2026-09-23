# torchbvh

GPU-native geometry primitives for PyTorch workflows: BVH construction, exact k-NN search, closest-hit ray tracing, MLS interpolation, displaced-query helpers, and FPS.

## Performance

The production routes use cooperative BVH construction, cached-bound k-NN, indexed packed MLS, bucketed FPS, and specialized segment/general triangle ray traversal.
Maintained benchmark commands are documented in [Performance](docs/performance.md).


## Install

Install CUDA-enabled PyTorch for your GPU, the matching CUDA toolkit (including NVCC), and a host C++ compiler supported by that toolkit. Then build `torchbvh` against the PyTorch already installed in your environment:

```bash
python -m pip install --upgrade setuptools wheel
python -m pip install --no-build-isolation --no-binary torchbvh torchbvh
```

`torchbvh` is **CUDA-only**. Version 0.3.0 is distributed as source rather than as a GPU-specific wheel. By default, PyTorch compiles for the GPUs visible during the build; set `TORCH_CUDA_ARCH_LIST` before installation when building for a different or broader set of GPU architectures.
See the [testing and build guide](docs/testing.md) for checks and compiler notes.

## Docs

Documentation can be found at [torchbvh.readthedocs.io](https://torchbvh.readthedocs.io/).

## Quickstart

```python
import torch
import torchbvh as tb

points = torch.randn(128, 3, device="cuda")
features = torch.randn(128, 8, device="cuda", requires_grad=True)

# A BVH owns its resources; the context manager releases them on exit.
with tb.BVH(points) as bvh:
    indices, dist_sq = bvh.knn(points[:16], k=8)  # (16, 8) each
    values = bvh.interpolate(points[:16], features, k=8)  # (16, 8)

values.sum().backward()  # MLS gradients flow to features

# Exact farthest-point sampling and assignment metadata.
samples = tb.fps(points, target_tokens=32)
samples.points           # (32, 3)
samples.nearest_anchor   # (128,)

# The same BVH class accepts fixed-size batches.
batched_points = torch.randn(2, 32, 3, device="cuda")
with tb.BVH(batched_points) as bvh:
    batch_indices, batch_dist_sq = bvh.knn(batched_points[:, :8], k=4)
# Both tensors have shape (2, 8, 4).
```

Continuing from the imports above, a 2-D segment-ray result can route each
query to one of two MLS feature fields. Misses use the field branch:

```python
segments = torch.rand(2, 16, 2, 2, device="cuda")  # (B, F, endpoints, D)
origins = torch.rand(2, 8, 2, 2, device="cuda")     # (B, M, H, D)
directions = torch.rand_like(origins) - 0.5
hits = tb.raytrace(segments, origins, directions,
                   primitive_type="segment", t_max=1.0)

boundary_pos = segments.reshape(2, 32, 2)
boundary_features = torch.rand(2, 32, 2, 4, device="cuda")
field_pos = torch.rand(2, 32, 2, device="cuda")
field_features = torch.rand(2, 32, 2, 4, device="cuda")
sampled = tb.conditional_mls_interpolate(
    hits.mask,
    true_points=boundary_pos,
    true_queries=hits.points,
    true_features=boundary_features,
    false_points=field_pos,
    false_queries=origins + directions,
    false_features=field_features,
    k=4,
)
# sampled has shape (2, 8, 2, 4).
```

Supports `D in {2, 3}`, `k in {4, 8, 16}`, and CUDA float32 inputs.


## References
`torchbvh` builds an implicit bounding volume hierarchy over 2-D or 3-D points.
The BVH layout follows the ostensibly-implicit tree formulation of Chitalu, Dubach, and Komura, and the Python/CUDA implementation was ported from the Julia `ImplicitBVH.jl` implementation.

- Floyd M. Chitalu, Christophe Dubach, and Taku Komura. "Binary Ostensibly-Implicit Trees for Fast Collision Detection." Computer Graphics Forum, 39(2), 509-521, 2020. DOI: [10.1111/cgf.13948](https://doi.org/10.1111/cgf.13948).
- `ImplicitBVH.jl`, StellaOrg. Julia implementation of the implicitly indexed BVH formulation from which the `torchbvh` BVH code was ported: [github.com/StellaOrg/ImplicitBVH.jl](https://github.com/StellaOrg/ImplicitBVH.jl).
