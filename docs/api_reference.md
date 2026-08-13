# API Reference

This page is a manual reference for the stable public `torchbvh` API. It does not
use autodoc or import the CUDA extension during documentation builds.

All geometry APIs expect finite CUDA `float32` tensors unless a section states
otherwise. Public APIs accept non-contiguous PyTorch views and may make
layout-only contiguous copies internally for CUDA kernels. Supported dimensions
are `D in {2, 3}` and supported k-NN sizes are `k in {4, 8, 16}`. Neighbor
distances are squared Euclidean distances sorted in ascending order.

## Class-Based API

### `BVH`

```python
BVH(points: torch.Tensor, *, batch_offsets: torch.Tensor | None = None)
```

Unified Python-owned BVH wrapper.

Input forms:

- `points: (N, D)` builds one single-sample BVH.
- `points: (B, N, D)` builds a fixed-size batched BVH.
- `points: (total_N, D)` with `batch_offsets: (B + 1,)` builds a packed ragged
  BVH.

`batch_offsets` must be CUDA `int64`, start at `0`, end at `total_N`, and be
strictly increasing. Ragged samples must each contain at least `k` points for the
later query.

Methods and properties:

```python
bvh.destroyed -> bool
bvh.knn(query_points, k, *, query_offsets=None, source_points=None, sort_queries=True)
bvh.interpolate(displaced_points, features, k=8, *, return_grad=False)
bvh.destroy() -> None
```

`knn(...)` dispatches to `query_knn(...)` using the owned handle. Single-sample
queries use `query_points: (M, D)` and return `(M, k)` tensors. Fixed-size
batched queries use `query_points: (B, M, D)` and return `(B, M, k)` tensors.
Ragged queries require `query_offsets: (B + 1,)` and return packed
`(total_M, k)` tensors. Indices are local to each source sample.

When `source_points` is provided, `knn(...)` returns
`(indices, squared_distances, neighbor_positions)`. Neighbor positions have shape
`(M, k, D)`, `(B, M, k, D)`, or `(total_M, k, D)` for single, batched, and
ragged queries.

`interpolate(...)` runs MLS interpolation for single and fixed-size batched BVHs.
It raises `TypeError` for ragged BVHs. Use the same shapes as
`mls_interpolate(...)`.

`BVH` supports context-manager use. Leaving the `with` block calls `destroy()`.
`destroy()` is idempotent. Calling methods after destruction raises
`RuntimeError`.

### `BatchedBVH`

```python
BatchedBVH(points: torch.Tensor)
```

Convenience subclass for fixed-size batches. `points` must have shape
`(B, N, D)`. It has the same methods and lifecycle behavior as `BVH(points)`.
Passing a non-3-D tensor raises `ValueError`.

### `RaggedBVH`

```python
RaggedBVH(points: torch.Tensor, batch_offsets: torch.Tensor)
```

Convenience subclass for packed variable-size batches. Equivalent to
`BVH(points, batch_offsets=batch_offsets)`. It supports `knn(...)` with
`query_offsets` and lifecycle methods. `interpolate(...)` raises `TypeError`.

### `RayBVH` and `raytrace`

```python
RayBVH(primitives, *, primitive_type="segment" | "triangle")
bvh.trace(origins, directions, *, t_min=1e-7, t_max=float("inf"))
raytrace(primitives, origins, directions, *, primitive_type, t_min=1e-7, t_max=float("inf"))
```

Segments have shape `(F,2,2)` or `(B,F,2,2)` and triangles have shape
`(F,3,3)` or `(B,F,3,3)`. Single-BVH rays may have shape `(...,D)`; batched
rays use `(B,...,D)`. Fixed-size batches build one independent BVH per sample,
while every token/head ray in that sample traverses the same BVH.

Both calls return `RayHitResult(primitive_indices, t, points, mask)`. Misses use
index `-1`, `t=inf`, `points=nan`, and `mask=False`. Directions need not be
normalized. Bounds are scalar and inclusive. Triangles are double-sided.
`RayBVH` supports `.destroyed`, idempotent `.destroy()`, and context-manager use;
geometry must not change while its BVH is reused.

## Handles And Lifecycle

### `BVHHandle`, `BatchedBVHHandle`, `RaggedBVHHandle`

Builder functions return Python handle objects that own tensor payload
references used by query functions.

Handle behavior:

- Handles expose `destroyed -> bool`.
- `destroy_bvh(handle)` or `handle.destroy()` releases Python tensor references.
- Destroying a handle more than once is allowed.
- Reading a destroyed handle or querying with it raises `RuntimeError`.
- Passing the wrong handle variant to a per-variant query raises `TypeError`.

Handles are mapping-like compatibility objects for low-level procedural code.
Class wrappers such as `BVH` intentionally do not expose mapping access.

`RaggedBVHHandle` owns per-sample inner handles. Destroying it cascades to those
inner handles.

### `destroy_bvh`

```python
destroy_bvh(bvh: BVHHandle | BatchedBVHHandle | RaggedBVHHandle | dict) -> None
```

Destroys a handle or legacy plain dictionary returned by older builder paths.
Unsupported objects raise `TypeError`.

## Build And Query Functions

### `build_bvh`

```python
build_bvh(points: torch.Tensor, *, batch_offsets: torch.Tensor | None = None)
    -> BVHHandle | BatchedBVHHandle | RaggedBVHHandle
```

Unified builder. It dispatches by input rank and `batch_offsets`:

- `(N, D)` returns `BVHHandle`.
- `(B, N, D)` returns `BatchedBVHHandle`.
- `(total_N, D)` plus `batch_offsets` returns `RaggedBVHHandle`.

Invalid ranks, dimensions, dtype, device, or ragged shape mismatches raise
`ValueError`.

### `build_bvh_batched`

```python
build_bvh_batched(points: torch.Tensor) -> BatchedBVHHandle
```

Backward-compatible fixed-size batched builder. `points` must be
`(B, N, D)`.

### `build_bvh_ragged`

```python
build_bvh_ragged(points: torch.Tensor, batch_offsets: torch.Tensor) -> RaggedBVHHandle
```

Backward-compatible ragged builder. `points` is `(total_N, D)` and
`batch_offsets` is `(B + 1,)`. Ragged neighbor indices are local within each
sample, not global packed-row indices.

### `query_knn`

```python
query_knn(
    bvh,
    query_points: torch.Tensor,
    k: int,
    *,
    query_offsets: torch.Tensor | None = None,
    source_points: torch.Tensor | None = None,
    sort_queries: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

Unified exact k-NN query. It dispatches by handle type.

Return values:

- Without `source_points`: `(indices, squared_distances)`.
- With `source_points`: `(indices, squared_distances, neighbor_positions)`.

Shape contracts:

| Handle | Query shape | Required offsets | Return index/dist shape | Neighbor position shape |
| --- | --- | --- | --- | --- |
| `BVHHandle` | `(M, D)` | none | `(M, k)` | `(M, k, D)` |
| `BatchedBVHHandle` | `(B, M, D)` | none | `(B, M, k)` | `(B, M, k, D)` |
| `RaggedBVHHandle` | `(total_M, D)` | `query_offsets: (B + 1,)` | `(total_M, k)` | `(total_M, k, D)` |

`indices` are `int64` local source indices. `squared_distances` are `float32`.
For ragged queries, `query_offsets` must match the ragged source batch count.
Each ragged source sample must contain at least `k` points.

`source_points`, when provided, must match the source geometry shape and device.
It is used only for gathering returned neighbor positions.

`sort_queries=True` uses the Morton-sorted traversal path for single and
fixed-size batched handles. Ragged queries internally use the single-sample
query path per sample.

Exception boundaries:

- Unsupported `k` raises `ValueError`.
- Bad ranks, dimensions, dtype, device, offsets, or count mismatches raise
  `ValueError`.
- Wrong handle variants raise `TypeError`.
- Destroyed handles raise `RuntimeError`.

### `query_knn_batched`

```python
query_knn_batched(
    bvh: BatchedBVHHandle,
    query_points: torch.Tensor,
    k: int,
    *,
    source_points: torch.Tensor | None = None,
    sort_queries: bool = True,
)
```

Backward-compatible fixed-size batched query. Shapes and returns match the
`BatchedBVHHandle` row in `query_knn(...)`.

### `query_knn_ragged`

```python
query_knn_ragged(
    bvh: RaggedBVHHandle,
    query_points: torch.Tensor,
    query_offsets: torch.Tensor,
    k: int,
    *,
    source_points: torch.Tensor | None = None,
)
```

Backward-compatible ragged query. Shapes and returns match the
`RaggedBVHHandle` row in `query_knn(...)`.

## MLS Interpolation

### `mls_interpolate`

```python
mls_interpolate(
    points: torch.Tensor,
    displaced_points: torch.Tensor,
    features: torch.Tensor,
    k: int = 8,
    *,
    return_grad: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]
```

Unified MLS interpolation. Dispatches by `points.ndim`.

Single-sample shapes:

- `points: (N, D)`
- `displaced_points: (M, D)`
- `features: (N, C)`
- return: `(M, C)`
- with `return_grad=True`: `((M, C), (M, D, C))`

Fixed-size batched shapes:

- `points: (B, N, D)`
- `displaced_points: (B, M, D)`
- `features: (B, N, C)`
- return: `(B, M, C)`
- with `return_grad=True`: `((B, M, C), (B, M, D, C))`

MLS requires `N >= k`, matching batch sizes and dimensions, and CUDA `float32`
inputs on the same device. Non-contiguous `points`, `displaced_points`, and
`features` are accepted and copied to contiguous layout internally only when
needed.

BVH construction and discrete neighbor selection are detached. PyTorch gradients
flow through the MLS solve to `features` and `displaced_points`. They do not flow
through BVH construction, neighbor indices, squared distances, or gathered
neighbor positions.

`return_grad=True` returns a spatial field-gradient tensor. It is ordinary
operator output, not PyTorch autograd metadata. `return_grad=False` returns only
the interpolated tensor.

Unsupported `k` and bad input contracts raise `ValueError`.

### `bvh_mls_interpolate`

```python
bvh_mls_interpolate(points, displaced_points, features, k=8, *, return_grad=False)
```

Backward-compatible single-sample MLS function. Shapes and returns match the
single-sample `mls_interpolate(...)` contract.

### `bvh_mls_interpolate_batched`

```python
bvh_mls_interpolate_batched(points, displaced_points, features, k=8, *, return_grad=False)
```

Backward-compatible fixed-size batched MLS function. Shapes and returns match
the batched `mls_interpolate(...)` contract.

### `conditional_mls_interpolate`

```python
conditional_mls_interpolate(
    mask,
    *,
    true_points,
    true_queries,
    true_features,
    false_points,
    false_queries,
    false_features,
    k=4,
    return_grad=False,
)
```

Evaluates exactly one matching-head MLS branch for each query:

```python
output[b, m, h] = (
    MLS(true_points, true_queries, true_features)
    if mask[b, m, h]
    else MLS(false_points, false_queries, false_features)
)
```

Input and output shapes:

- `mask: (B, M, H)` CUDA `bool`.
- branch points: `(B, N_branch, D)` CUDA `float32`.
- branch queries: `(B, M, H, D)` CUDA `float32`.
- branch features: `(B, N_branch, H, C)` CUDA `float32`.
- output: `(B, M, H, C)`.
- with `return_grad=True`: output plus spatial field gradient
  `(B, M, H, D, C)`.

The branches must share `B`, `M`, `H`, `D`, `C`, dtype, and device, but
`N_true` and `N_false` may differ. Each source count must be at least `k`.
Inactive query rows are the explicit exception to the finite-input rule: they
may contain NaNs because `torch.where` excludes them before BVH traversal.

The operation builds two temporary point BVHs, route-sorts the selected queries,
and performs one exact k-NN traversal and one MLS solve per query. Source
positions and discrete selection are detached. Gradients flow to selected query
rows and active source features; inactive query rows and inactive feature rows
receive zero gradient. `mask` is non-differentiable.

A boundary/field workflow keeps closest-hit ray tracing separate:

```python
hits = boundary_bvh.trace(origins, displacements, t_max=1.0)
values = torchbvh.conditional_mls_interpolate(
    hits.mask,
    true_points=boundary_pos,
    true_queries=hits.points,
    true_features=boundary_features,
    false_points=field_pos,
    false_queries=origins + displacements,
    false_features=field_features,
    k=4,
)
```

## Displaced-Query Helpers

These helpers are for fixed-size batched, matching-head displaced queries. They
build one BVH per batch sample over `pos` and flatten only query heads.

### `query_displaced_knn`

```python
query_displaced_knn(
    pos: torch.Tensor,
    q: torch.Tensor,
    k: int = 8,
    *,
    return_positions: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

Inputs:

- `pos: (B, N, D)` CUDA `float32` source positions.
- `q: (B, N, H, D)` CUDA `float32` displaced query positions.
- `H >= 1`.

Returns:

- `indices: (B, N, H, k)` local source indices.
- `squared_distances: (B, N, H, k)`.
- `neighbor_positions: (B, N, H, k, D)` when `return_positions=True`.

Outputs are detached from `pos` and `q`. Unsupported `k` or bad shapes, dtype,
or device raise `ValueError`.

### `gather_neighbor_values`

```python
gather_neighbor_values(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor
```

Gathers matching-head values from k-NN indices.

- `values: (B, N, H, Ch)` CUDA `float32`.
- `indices: (B, N, H, k)` CUDA `int64` local source indices.
- return: `(B, N, H, k, Ch)`.

Indices must be in `[0, N)`. Gradients flow to `values` only. Contract
violations raise `ValueError`.

### `interpolate_displaced`

```python
interpolate_displaced(
    pos: torch.Tensor,
    q: torch.Tensor,
    values: torch.Tensor,
    k: int = 8,
    *,
    reduction: str = "weighted_mean",
) -> torch.Tensor
```

Interpolates matching-head values at displaced queries.

Inputs follow `query_displaced_knn(...)` plus `values: (B, N, H, Ch)`. The return
shape is `(B, N, H, Ch)`.

The only supported reduction is `"weighted_mean"`. It uses inverse
squared-distance weights. Exact hits use the unweighted mean over exact-hit
neighbors. Any other reduction raises `ValueError`.

Gradients flow to `values` only. They do not flow to `pos`, `q`, indices, or
distances.

## FPS

### `fps`

```python
fps(
    points: torch.Tensor,
    target_tokens: int,
    *,
    seed: int = 0,
    r: int = 4,
    c: int = 2,
    alpha: float = 0.25,
    mode: str = "exact_bucketed",
    bucket_size: int = 256,
    use_graph: bool = True,
) -> FPSResult
```

Farthest point sampling geometry for CUDA point clouds.

Input shapes:

- Single sample: `points: (N, D)`.
- Fixed-size batch: `points: (B, N, D)`.

`target_tokens` must be in `[1, N]`. `seed` must be `-1` or an original point
index in `[0, N)`. `seed=-1` chooses the point nearest the input AABB center.

Modes:

- `"exact_bucketed"`: default exact path.
- `"exact_full_scan"`: exact full-scan fallback.
- `"approx_bucketed"`: opt-in approximate bucket-queue path.

For `"approx_bucketed"`, `r` must be in `[1, 8]`, `c >= 1`, and `r * c <= 32`.
`alpha` must be nonnegative. Invalid mode or knob values raise `ValueError`.

FPS is non-differentiable. It returns geometry and assignment metadata for use in
user-owned pooling or unpooling logic.

### `FPSResult`

```python
@dataclass
class FPSResult:
    indices: torch.Tensor
    points: torch.Tensor
    nearest_anchor: torch.Tensor
    nearest_anchor_dist_sq: torch.Tensor
    anchor_radius: torch.Tensor
    anchor_counts: torch.Tensor
    coarse_order: torch.Tensor
    selection_order_indices: torch.Tensor
```

Single-sample result shapes:

| Field | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `indices` | `(M,)` | `int64` | selected original source indices in selection order |
| `points` | `(M, D)` | `float32` | selected anchor points |
| `nearest_anchor` | `(N,)` | `int32` | selection-order anchor assignment for each source point |
| `nearest_anchor_dist_sq` | `(N,)` | `float32` | squared distance to assigned anchor |
| `anchor_radius` | `(M,)` | `float32` | max assigned squared distance per anchor |
| `anchor_counts` | `(M,)` | `int32` | number of source points assigned per anchor |
| `coarse_order` | `(M,)` | `int64` | permutation ordering anchors by BVH leaf position |
| `selection_order_indices` | `(M,)` | `int64` | selected original indices in FPS selection order |

Batched result shapes add a leading `B`: `indices: (B, M)`, `points:
(B, M, D)`, assignment fields over source points as `(B, N)`, and anchor fields
as `(B, M)`.

`nearest_anchor` indexes anchors in selection order, not Morton order. To gather
anchors in Morton order, first gather with `coarse_order` and remap assignments
accordingly in user code.

## Constants

```python
SUPPORTED_K = (4, 8, 16)
SUPPORTED_DIMS = (2, 3)
```

These constants record the public supported k-NN sizes and geometry dimensions.

## Debug Helpers

The package also exports small arithmetic/debug symbols used by tests:
`implicit_tree_ancestor`, `implicit_tree_descendant`, `implicit_tree_summary`,
`morton_encode_2d`, `morton_encode_3d`, `morton_split2`, `morton_split3`, and
`smoke_add_one`. They are not user-facing workflow APIs.
