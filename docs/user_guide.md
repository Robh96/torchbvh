# User Guide

`torchbvh` expects CUDA float32 tensors, `D in {2, 3}`, and `k in {4, 8, 16}`.
Callers provide finite inputs. Public APIs accept non-contiguous PyTorch views
and may make layout-only contiguous copies internally for CUDA kernels.
Gradients do not flow through BVH construction, FPS, or discrete neighbor
selection.

## Single-Sample k-NN

Use `BVH(points)` for one point cloud. Distances are squared Euclidean distances
sorted ascending with their matching indices.

```python
import torch
import torchbvh as tb

points = torch.randn(4096, 3, device="cuda")                 # (N, D) float32
query_pts = points                                           # (N, D) float32

bvh = tb.BVH(points)
idx, dist_sq = bvh.knn(query_pts, k=8)                       # (N, 8) int64, (N, 8) float32
```

The class owns the native handle. Use the context-manager form when lifetime
should be scoped to a block.

```python
with tb.BVH(points) as bvh:                                  # points: (N, D)
    idx, dist_sq = bvh.knn(query_pts, k=8)                   # (N, 8), (N, 8)
```

## Fixed-Size Batched k-NN

For equal-size batches, pass `(B, N, D)` points. Query points use the same batch
dimension and return local per-sample indices.

```python
points = torch.randn(4, 4096, 3, device="cuda")              # (B, N, D) float32
query_pts = points                                           # (B, N, D) float32

bvh = tb.BVH(points)
idx, dist_sq = bvh.knn(query_pts, k=8)                       # (B, N, 8), (B, N, 8)
```

## Ragged k-NN

For variable-size samples, pack source points as `(total_N, D)` and provide
1-D int64 start offsets. Ragged handles support k-NN only.

```python
points = torch.randn(9000, 3, device="cuda")                 # (total_N, D) float32
batch_offsets = torch.tensor([0, 2000, 5500, 9000],           # (B + 1,) int64
                             device="cuda", dtype=torch.int64)
query_pts = torch.randn(7200, 3, device="cuda")              # (total_M, D) float32
query_offsets = torch.tensor([0, 1800, 4200, 7200],           # (B + 1,) int64
                             device="cuda", dtype=torch.int64)

bvh = tb.BVH(points, batch_offsets=batch_offsets)
idx, dist_sq = bvh.knn(query_pts, k=8, query_offsets=query_offsets)  # (total_M, 8), (total_M, 8)
```

`bvh.interpolate(...)` raises `TypeError` for ragged handles. Build a
single-sample or fixed-size batched BVH for MLS interpolation.

## MLS Interpolation

Use MLS when features live on source points and query positions may be displaced.
Gradients flow to `features` and live `displaced_pts`.

```python
points = torch.randn(4096, 3, device="cuda")                         # (N, D) float32
displaced_pts = points + 0.01 * torch.randn_like(points)              # (N, D) float32
displaced_pts.requires_grad_(True)
features = torch.randn(4096, 16, device="cuda", requires_grad=True)   # (N, Ch) float32

bvh = tb.BVH(points)
interpolated = bvh.interpolate(displaced_pts, features, k=8)          # (N, Ch) float32
```

The procedural alias is useful in low-level pipelines.

```python
interpolated = tb.mls_interpolate(points, displaced_pts, features, k=8) # (N, Ch) float32
```

`return_grad=True` returns the spatial MLS field gradient. It is data returned by
the operator, not PyTorch autograd metadata.

```python
interpolated, field_gradient = bvh.interpolate(
    displaced_pts, features, k=8, return_grad=True
)                                                                    # (N, Ch), (N, D, Ch)
```

The same calls support fixed-size batches with `(B, N, D)` geometry and
`(B, N, Ch)` features.

## Displaced-Query Workflow

For multihead displaced queries, build one BVH per sample over `pos` and reuse it
across all `H` heads. The helper owns that shape choreography.

```python
pos = torch.randn(2, 4096, 3, device="cuda")                    # (B, N, D) float32
rho = torch.randn(2, 4096, 4, 3, device="cuda")                 # (B, N, H, D) float32
displaced_queries = pos[:, :, None, :] + rho                    # (B, N, H, D) float32
values = torch.randn(2, 4096, 4, 32, device="cuda")             # (B, N, H, Ch) float32

idx, dist_sq = tb.query_displaced_knn(
    pos, displaced_queries, k=8, return_positions=False
)                                                               # (B, N, H, 8), (B, N, H, 8)
gathered = tb.gather_neighbor_values(values, idx)               # (B, N, H, 8, Ch)
out = tb.interpolate_displaced(pos, displaced_queries, values, k=8) # (B, N, H, Ch)
```

Gradients flow to `values` only through `gather_neighbor_values(...)` and
`interpolate_displaced(...)`. Indices, distances, source positions, and displaced
query geometry are discrete query inputs for these helpers.

## FPS Downsampling Geometry

`fps(points, target_tokens)` selects farthest-point anchors for `(N, D)` or
`(B, N, D)` inputs and returns geometry plus assignment metadata.

```python
points = torch.randn(4, 4096, 3, device="cuda")                 # (B, N, D) float32
result = tb.fps(points, target_tokens=1024)

result.indices                     # (B, M) int64, selected source indices
result.points                      # (B, M, D) float32, selected anchor points
result.nearest_anchor              # (B, N) int32, assignment to selection-order anchor
result.nearest_anchor_dist_sq      # (B, N) float32, squared distance to assigned anchor
result.anchor_radius               # (B, M) float32, max assigned squared distance
result.anchor_counts               # (B, M) int32, number of points assigned per anchor
result.coarse_order                # (B, M) int64, anchors ordered by BVH leaf position
```

For a single sample, remove the leading `B` dimension: `indices` is `(M,)`,
`points` is `(M, D)`, and assignment metadata is `(N,)` or `(M,)`.

FPS is non-differentiable. Pass the assignment metadata to your own PyTorch
pooling or unpooling logic.

## API Choice

Prefer `BVH` for new code. It owns handle lifetime and dispatches across
single-sample, fixed-size batched, and ragged k-NN workflows.

Use unified procedural functions such as `build_bvh`, `query_knn`, and
`mls_interpolate` when a low-level pipeline needs explicit handle control.

Per-variant aliases such as `build_bvh_batched` and `query_knn_batched` remain
stable for compatibility and shape-specific integrations, but they are not the
recommended starting point.
