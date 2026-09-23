# Lifecycle And Gradients

`torchbvh` handles are Python-owned tensor containers. Geometry construction, tree metadata, traversal, FPS selection, and integer neighbor choices are non-differentiable. Gradients flow only through the explicit tensor operations called out below.

Users must provide finite input tensors. The package does not promise a `torch.isfinite` scan or defined behavior for NaN/Inf values.

`conditional_mls_interpolate` has one narrow exception: query rows inactive under its boolean mask may contain NaNs. The live selected query is formed before traversal, so inactive rows are never encoded, searched, or solved.

## Handle Lifetime

Prefer `BVH` for new code. It owns the underlying handle, dispatches across single-sample, fixed-size batched, and ragged k-NN inputs, and hides internal mapping fields.

```python
with tb.BVH(points) as bvh:
    idx, dist_sq = bvh.knn(query_points, k=8)
```

`BVH` supports `.destroyed`, `.destroy()`, and context-manager use. Leaving a `with` block calls `.destroy()`. Calling `.destroy()` more than once is safe.

`RayBVH` has the same lifecycle contract but owns a primitive BVH rather than a point BVH. Its geometry is immutable for the handle lifetime; rebuild after an optimizer update or any in-place geometry change.

Procedural code can use handles directly:

- `BVHHandle` from `build_bvh(points)` for `(N, D)`.
- `BatchedBVHHandle` from `build_bvh(points)` for `(B, N, D)`.
- `RaggedBVHHandle` from `build_bvh(points, batch_offsets=...)` for packed ragged inputs.

`destroy_bvh(handle)` delegates to the handle's `.destroy()` method. Destroying a handle clears its Python tensor references; there is no native handle registry or extra native teardown step.

After destruction, mapping access or query use raises `RuntimeError`. This is the destroyed-handle boundary. Idempotent destroy still succeeds:

```python
handle = tb.build_bvh(points)
tb.destroy_bvh(handle)
tb.destroy_bvh(handle)       # allowed
tb.query_knn(handle, q, 8)   # RuntimeError
```

`RaggedBVHHandle` owns inner per-sample `BVHHandle` objects. Destroying the ragged handle cascades to those inner handles, then clears the outer metadata.

Plain dictionaries from older builder paths are no longer accepted by `query_knn(...)` or `destroy_bvh(...)`. Keep the returned handle object or use `BVH`; do not rely on internal marker keys.

## Exception Boundaries

`TypeError` means the handle type is wrong or `query_offsets` was supplied for a non-ragged handle.

`ValueError` means Python-side semantic validation failed: unsupported `k`, bad offsets, invalid shapes in validated wrappers, device mismatches, dtype or contiguity checks, unsupported displaced-query reductions, or similar input contract errors.

`RuntimeError` means a destroyed handle was accessed, or a native structural check failed. Native build paths can raise `RuntimeError` for rank, dtype, device, contiguity, or dimension errors.

## Non-Differentiable Boundaries

The following are metadata or discrete geometry decisions, not differentiable PyTorch computations:

- BVH construction.
- Morton sorting.
- tree topology and handle metadata/lifecycle.
- k-NN traversal and discrete neighbor selection.
- integer indices.
- query squared distances.
- FPS anchor selection.
- FPS assignment metadata, including nearest-anchor ids, radii, counts, and ordering metadata.

Production MLS uses its indexed packed autograd kernel directly. The old detached query classes were removed in 0.3.0.

## Where Gradients Flow

MLS interpolation (`mls_interpolate`, the matching-head API, and `BVH.interpolate`) detaches BVH construction and neighbor selection, then runs the MLS solve on live tensors. Gradients flow to:

- `features`;
- live `displaced_points`.

Gradients do not flow to the source `points` passed to MLS wrappers.

`conditional_mls_interpolate` follows the same boundary for both source point sets. `torch.where` routes gradients to the selected query branch, and concatenating feature banks routes feature gradients to rows used by the active branch. Inactive query rows and inactive feature-bank rows receive zero gradient. The mask, source positions, route/Morton ordering, neighbor indices, and squared distances are non-differentiable.

`return_grad=True` returns `(interpolated, field_gradient)`. The `field_gradient` tensor is the spatial derivative of the interpolated field. It is ordinary operator output, not PyTorch autograd metadata, and enabling it does not make BVH construction or neighbor selection differentiable.

`gather_neighbor_values(values, indices)` propagates gradients to `values` only. `indices` are integer selection metadata.

`interpolate_displaced(pos, q, values, ...)` propagates gradients to `values` only. Gradients do not flow to `pos`, `q`, indices, or squared distances.

Direct `query_knn` calls have one accepted asymmetry: when explicit `source_points` are provided, the optional returned `neighbor_positions` are gathered with ordinary PyTorch indexing over already-selected integer indices. That gather can propagate gradients to the explicit `source_points` tensor only.
It is not a gradient through BVH construction, traversal, query coordinates, or neighbor selection.

Ray tracing similarly detaches BVH construction and the winning primitive ID. The segment path uses a fused native analytic backward; the triangle path gathers the selected live primitive and reconstructs the analytic intersection with PyTorch operations. `RayHitResult.t` and `.points` propagate piecewise gradients to origins, directions, and the selected segment/triangle vertices.
Gradients do not cross a hit-selection boundary and miss outputs have zero gradients.

The fused MLS and segment-ray custom autograd operations support first-order gradients. Higher-order differentiation through their backward formulas is not part of the public contract.

## Where Gradients Do Not Flow

No PyTorch gradients are produced for:

- source points in MLS wrappers;
- `pos` or `q` in `query_displaced_knn(...)` and `interpolate_displaced(...)`;
- k-NN indices and squared distances;
- FPS outputs and `FPSResult` metadata;
- handle metadata, lifecycle state, or destroyed-handle checks.
