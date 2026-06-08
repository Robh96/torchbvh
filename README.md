# torchbvh

GPU-native geometry primitives for PyTorch point-cloud workflows: BVH construction,
exact k-NN search, MLS interpolation, displaced-query helpers, and FPS downsampling.

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

See `benchmarks/third_party_algorithm_comparison.ipynb` for more detailed comparisons.


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
idx, dists = bvh.knn(points, k=8)         # (N,8) int64, (N,8) float32

# MLS interpolation — gradients flow through features
feat = torch.randn(1024, 16, device="cuda", requires_grad=True)
out = bvh.interpolate(points, feat, k=8)  # (N, 16)

# FPS downsampling geometry
fps = tb.fps(points, target_tokens=256)
# fps.indices, fps.points, fps.nearest_anchor, fps.anchor_radius, ...

# Batched: pass (B, N, D) → returns (B, N, k)
```

Supports `D in {2, 3}`, `k in {4, 8, 16}`, CUDA float32 contiguous inputs.


## References
`torchbvh` builds an implicit bounding volume hierarchy over 2-D or 3-D points.
The BVH layout follows the ostensibly-implicit tree formulation of Chitalu, Dubach, and
Komura, and the Python/CUDA implementation was ported from the Julia
`ImplicitBVH.jl` implementation.

- Floyd M. Chitalu, Christophe Dubach, and Taku Komura. "Binary
  Ostensibly-Implicit Trees for Fast Collision Detection." Computer Graphics Forum,
  39(2), 509-521, 2020. DOI:
  [10.1111/cgf.13948](https://doi.org/10.1111/cgf.13948).
- `ImplicitBVH.jl`, StellaOrg. Julia implementation of the implicitly indexed BVH
  formulation from which the `torchbvh` BVH code was ported:
  [github.com/StellaOrg/ImplicitBVH.jl](https://github.com/StellaOrg/ImplicitBVH.jl).
