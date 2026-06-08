# Algorithms

This page describes the public algorithms behind `torchbvh` operations. It focuses on the algorithm steps and observable behavior, not CUDA implementation details.

## BVH Construction

`torchbvh` builds an implicit bounding volume hierarchy over 2-D or 3-D points.
The BVH layout follows the ostensibly-implicit tree formulation of Chitalu, Dubach, and
Komura, and the Python/CUDA implementation was ported from the Julia
`ImplicitBVH.jl` implementation.

1. Compute the scene axis-aligned bounding box for the input points.
2. Normalize each point into that scene box and assign it a Morton code.
3. Sort source point indices by Morton code. The source point tensor itself stays in original order.
4. Treat the sorted points as leaves of an implicit binary tree. If the leaf count is not a power of two, virtual leaves fill the rightmost missing positions.
5. Store each real leaf's AABB as the point coordinate repeated as lower and upper bounds.
6. Build internal node AABBs bottom-up by merging real child AABBs. If an internal node has only one real child, its AABB equals that child's AABB.
7. Keep `sorted_indices` so leaf positions can map back to original source indices.

Fixed-size batched BVHs repeat the same process independently for each sample in the batch. Ragged BVHs apply the same single-sample process to each packed segment described by `batch_offsets`.

Why it is fast: The construction avoids serial tree insertion and pointer-heavy node allocation. Morton sorting converts spatial hierarchy construction into a parallel sort-plus-reduction problem, and the compact implicit tree stores only real node AABBs
and source-index mappings. The main bottleneck in pointer or CPU tree builders is irregular allocation and recursive dependency; this design replaces it with contiguous tensor work that can be built and traversed with predictable memory access.

## k-NN Query

k-NN is an exact branch-and-bound search over the BVH. Returned distances are squared Euclidean distances sorted from nearest to farthest.

For each query point:

1. Start with an empty sorted neighbor list of length `k`, initialized to infinite distances.
2. Start traversal at the BVH root.
3. For a candidate node, compute the minimum possible squared distance from the query point to that node's AABB.
4. If that minimum distance is greater than the current kth-best distance, skip the whole node.
5. If the node is a leaf, compute the exact squared distance to its source point and insert the point into the sorted neighbor list if it improves the current list.
6. If the node is internal, evaluate both real children and visit the closer child first.
7. Continue until no reachable node can improve the neighbor list.
8. Return original source indices and their matching squared distances.

Self-neighbors are included when source points are queried against themselves. When two or more points are exactly tied, the returned tied indices are valid exact neighbors, but their order is not part of the public contract.

For fixed-size batches, each sample is searched independently and indices are local to that sample. For ragged batches, each packed segment is searched independently and indices are local to the segment, not global packed-row indices.

Why it is fast: Brute-force k-NN evaluates every query against every source point, so its distance-work scales as `M * N`. BVH traversal replaces most of those exact point distance evaluations with cheap AABB lower-bound tests. Once a query has `k` good neighbors, any subtree whose nearest possible point is farther than the current k-th neighbor is skipped entirely. The remaining per-query work is independent, so many queries can run in parallel without CPU-side tree traversal or Python loops.

## MLS Interpolation

Moving least squares interpolation fits a local linear field around each query position using k-NN neighborhoods.

For each displaced query point:

1. Run exact k-NN against the source geometry to get neighbor indices, squared distances, and neighbor positions. This selection is discrete and non-differentiable.
2. Gather neighbor features with the returned indices.
3. If one or more neighbors have squared distance at or below the exact-hit epsilon, use the unweighted mean of those exact-hit neighbor features as the interpolated value and return a zero spatial field gradient.
4. Otherwise, compute offsets from the query to each neighbor position.
5. Choose a local bandwidth from the sorted neighbor distances and clamp it away from zero.
6. Assign each neighbor a smooth weight from its squared offset distance and the local bandwidth.
7. Fit a regularized weighted linear model in local coordinates:
   value = constant term + spatial slope dot offset.
8. Return the fitted constant term as the interpolated feature.
9. When `return_grad=True`, return the fitted spatial slope as the field gradient.

Gradients flow through the MLS solve to `features` and live `displaced_points`.
Gradients do not flow through BVH construction, Morton sorting, k-NN selection, integer indices, squared distances, or detached neighbor positions.

Why it is fast: a dense differentiable interpolation would either compare each query to all source points or build large intermediate tensors for weights and gradients. MLS uses the exact k-NN result to restrict the solve to `k` local samples, then solves only a
small regularized linear system per query. The discrete geometry search is detached, so autograd tracks the continuous MLS solve for `features` and `displaced_points` without recording the BVH traversal, sorting, or integer neighbor selection.

## FPS

Farthest point sampling selects anchor points that cover the source geometry and returns assignment metadata for the selected anchors.

All FPS modes maintain this state:

1. `indices`: selected source point indices in selection order.
2. `nearest_anchor`: for each source point, the selection-order anchor currently closest to it.
3. `nearest_anchor_dist_sq`: the squared distance from each source point to that nearest anchor.

### Exact FPS

The exact modes follow the standard farthest point sampling rule.

1. Choose the first anchor from `seed`. If `seed=-1`, choose the source point nearest the input AABB center.
2. Initialize every source point's nearest-anchor distance to its squared distance from the first anchor.
3. Select the next anchor as a point with maximum current nearest-anchor distance.
4. Update every source point: if the new anchor is closer than its current nearest anchor, replace the stored distance and assignment.
5. Repeat selection and update until `target_tokens` anchors have been selected.
6. Gather selected anchor coordinates.
7. Compute per-anchor assignment counts and per-anchor radius from the final nearest-anchor assignments.
8. Compute `coarse_order`, which orders selected anchors by BVH leaf position for locality-aware downstream gathering.

`mode="exact_full_scan"` expresses this rule directly. `mode="exact_bucketed"` preserves the same exact result while using BVH/Morton bucket structure to organize the work.

Why it is fast: standard exact FPS is dominated by the repeated global update and maximum search over all `N` points for each of `M` anchors. `exact_full_scan` keeps that state on the GPU and avoids CPU round trips. `exact_bucketed` keeps the exact selection rule but groups points by BVH/Morton buckets, so bucket maxima identify farthest candidates and AABB-based pruning can skip bucket refreshes that cannot improve any point's current nearest-anchor distance. This reduces memory traffic while preserving the exact FPS sequence.

### Approximate Bucketed FPS

`mode="approx_bucketed"` keeps the same output metadata but relaxes the anchor selection rule for speed.

1. Build the same BVH/Morton spatial structure used by the exact bucketed path.
2. Choose the first anchor with the same seed policy.
3. Maintain exact nearest-anchor assignments and distances for the anchors that have already been selected.
4. Track candidate farthest points from spatial buckets instead of scanning the full point set for one global maximum every round.
5. Generate a small candidate set, reject duplicates, and commit one or more anchors per round according to the `r`, `c`, and `alpha` settings.
6. After committed anchors are chosen, update every affected source point's exact nearest-anchor distance and assignment against those committed anchors.
7. Refresh bucket state and repeat until `target_tokens` anchors have been selected.
8. Compute the same public metadata as exact FPS.

The approximate mode is useful when throughput is more important than matching the exact serial FPS sequence. Its returned assignments are exact with respect to the anchors it selected.

Why it is fast: the fundamental bottleneck in exact FPS is the serial dependency that selects one anchor, updates all distances, then selects the next anchor. The approximate bucketed mode breaks part of that dependency by proposing farthest candidates from spatial buckets and committing multiple anchors per round. Distance assignments are then updated against the committed anchor set in one pass. The algorithm trades exact global anchor order for fewer full distance-update and bucket-refresh cycles, while keeping the final nearest-anchor metadata exact for the anchors it actually selected.

## References

- Floyd M. Chitalu, Christophe Dubach, and Taku Komura. "Binary
  Ostensibly-Implicit Trees for Fast Collision Detection." Computer Graphics Forum,
  39(2), 509-521, 2020. DOI:
  [10.1111/cgf.13948](https://doi.org/10.1111/cgf.13948).
- `ImplicitBVH.jl`, StellaOrg. Julia implementation of the implicitly indexed BVH
  formulation from which the `torchbvh` BVH code was ported:
  [github.com/StellaOrg/ImplicitBVH.jl](https://github.com/StellaOrg/ImplicitBVH.jl).