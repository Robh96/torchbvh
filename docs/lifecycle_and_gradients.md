# Lifecycle And Gradients

`torchbvh` handles are Python-owned tensor containers. Geometry construction,
tree metadata, traversal, FPS selection, and integer neighbor choices are
non-differentiable. Gradients flow only through the explicit tensor operations
called out below.

Users must provide finite input tensors. The package does not promise a
`torch.isfinite` scan or defined behavior for NaN/Inf values.

## Handle Lifetime

Prefer `BVH` for new code. It owns the underlying handle, dispatches across
single-sample, fixed-size batched, and ragged k-NN inputs, and hides internal
mapping fields.

```python
with tb.BVH(points) as bvh:
    idx, dist_sq = bvh.knn(query_points, k=8)
```

`BVH`, `BatchedBVH`, and `RaggedBVH` support `.destroyed`, `.destroy()`, and
context-manager use. Leaving a `with` block calls `.destroy()`. Calling
`.destroy()` more than once is safe.

Procedural code can use handles directly:

- `BVHHandle` from `build_bvh(points)` for `(N, D)`.
- `BatchedBVHHandle` from `build_bvh(points)` or `build_bvh_batched(points)`
  for `(B, N, D)`.
- `RaggedBVHHandle` from `build_bvh(points, batch_offsets=...)` or
  `build_bvh_ragged(points, batch_offsets)` for packed ragged inputs.

`destroy_bvh(handle)` delegates to the handle's `.destroy()` method. Destroying
a handle clears its Python tensor references; there is no native handle registry
or extra native teardown step.

After destruction, mapping access or query use raises `RuntimeError`. This is
the destroyed-handle boundary. Idempotent destroy still succeeds:

```python
handle = tb.build_bvh(points)
tb.destroy_bvh(handle)
tb.destroy_bvh(handle)       # allowed
tb.query_knn(handle, q, 8)   # RuntimeError
```

`RaggedBVHHandle` owns inner per-sample `BVHHandle` objects. Destroying the
ragged handle cascades to those inner handles, then clears the outer metadata.

## Legacy Dictionaries

Current builders return handle objects, but procedural `query_knn(...)` and
`destroy_bvh(...)` still accept legacy plain dictionaries produced by older
builder paths. This compatibility exists only at the procedural query/destroy
boundary.

For new code, use the handle classes or `BVH`. Do not rely on internal marker
keys such as `_batched`, `_ragged`, or `_destroyed`.

## Exception Boundaries

`TypeError` means the handle variant or object type is wrong, such as passing a
single-sample `BVHHandle` to `query_knn_batched(...)`.

`ValueError` means Python-side semantic validation failed: unsupported `k`, bad
offsets, invalid shapes in validated wrappers, device mismatches, dtype or
contiguity checks, unsupported displaced-query reductions, or similar input
contract errors.

`RuntimeError` means a destroyed handle was accessed, or a native structural
check failed. Native build paths can raise `RuntimeError` for rank, dtype,
device, contiguity, or dimension errors.

## Non-Differentiable Boundaries

The following are metadata or discrete geometry decisions, not differentiable
PyTorch computations:

- BVH construction.
- Morton sorting.
- tree topology and handle metadata/lifecycle.
- k-NN traversal and discrete neighbor selection.
- integer indices.
- query squared distances.
- FPS anchor selection.
- FPS assignment metadata, including nearest-anchor ids, radii, counts, and
  ordering metadata.

`BVHQuery` and `BatchedBVHQuery` are explicit autograd boundaries used by MLS.
They detach source geometry, query geometry, indices, squared distances, and
neighbor positions.

## Where Gradients Flow

MLS interpolation (`mls_interpolate`, `bvh_mls_interpolate`,
`bvh_mls_interpolate_batched`, and `BVH.interpolate`) detaches BVH construction
and neighbor selection, then runs the MLS solve on live tensors. Gradients flow
to:

- `features`;
- live `displaced_points`.

Gradients do not flow to the source `points` passed to MLS wrappers.

`return_grad=True` returns `(interpolated, field_gradient)`. The
`field_gradient` tensor is the spatial derivative of the interpolated field. It
is ordinary operator output, not PyTorch autograd metadata, and enabling it does
not make BVH construction or neighbor selection differentiable.

`gather_neighbor_values(values, indices)` propagates gradients to `values` only.
`indices` are integer selection metadata.

`interpolate_displaced(pos, q, values, ...)` propagates gradients to `values`
only. Gradients do not flow to `pos`, `q`, indices, or squared distances.

Direct `query_knn*` calls have one accepted asymmetry: when explicit
`source_points` are provided, the optional returned `neighbor_positions` are
gathered with ordinary PyTorch indexing over already-selected integer indices.
That gather can propagate gradients to the explicit `source_points` tensor only.
It is not a gradient through BVH construction, traversal, query coordinates, or
neighbor selection.

## Where Gradients Do Not Flow

No PyTorch gradients are produced for:

- source points in MLS wrappers;
- `pos` or `q` in `query_displaced_knn(...)` and `interpolate_displaced(...)`;
- `indices`, `squared_distances`, or `neighbor_positions` returned by detached
  `BVHQuery` and `BatchedBVHQuery`;
- FPS outputs and `FPSResult` metadata;
- handle metadata, lifecycle state, or destroyed-handle checks.
