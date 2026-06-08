# Testing and Benchmarking

This document contains the validation and performance plan for `torchbvh`.

## Local Test Commands

Activate the documented Python environment before building or testing:

```powershell
. "C:\Users\Rob.Hart-Villamil\Documents\python_envs\ml\Scripts\activate.ps1"
```

Editable installs should disable build isolation so the extension build can import the environment's installed PyTorch:

```powershell
python -m pip install -e . --no-build-isolation
```

Use pytest without its cache provider on the current Windows workspace to avoid transient cache-directory permission errors:

```powershell
python -m pytest -q -p no:cacheprovider
```

`pytest.ini` limits default collection to the in-repo `tests/` directory and excludes
local ignored comparison checkouts, build outputs, artifacts, and `pytest-cache-files-*`
workspace noise. Optional third-party algorithm comparisons belong in
`benchmarks/third_party_algorithm_comparison.ipynb`, where the notebook installs and
skips optional packages explicitly; they are not part of project test collection.

## Julia Fixture Validation

For Components 1-3, generate expected outputs from the vendored Julia implementation and compare against Python/CUDA outputs.

Julia executable:

```text
C:\Users\Rob.Hart-Villamil\Documents\python_envs\ml\julia_env\pyjuliapkg\install\bin\julia.exe
```

Do not assume `julia.exe` is on `PATH`.

Before generating fixtures, instantiate the vendored Julia project if its dependencies are missing:

```powershell
& "C:\Users\Rob.Hart-Villamil\Documents\python_envs\ml\julia_env\pyjuliapkg\install\bin\julia.exe" --project=ImplicitBVH.jl -e "using Pkg; Pkg.instantiate()"
```

If Julia fixtures are temporarily unavailable, Python tests may validate the documented formulas directly, but later milestones should restore fixture comparison before depending on the arithmetic in runtime BVH code.

For header-only arithmetic components, prefer a narrow Python test surface that mirrors the component directly through scalar pybind helpers. These tests should use independent Python reference formulas where practical, and should not introduce tensor APIs, kernels, or runtime handles just to validate arithmetic.

For later runtime tests that need compact tree ordering, prefer `implicit_tree_summary()`
as the Python-side oracle for `memory_index`, real-node ranges, and virtual-node
boundaries. This keeps runtime tests focused on their own behavior instead of
reimplementing implicit-tree arithmetic in every test file.

Fixture generation outline:

```julia
using ImplicitBVH
using JSON

for t in [1, 2, 3, 5, 8, 15, 16, 17, 100, 1000, 1024, 1025]
    tree = ImplicitTree(t)
    data = Dict(
        "t" => t,
        "real_nodes" => tree.real_nodes,
        "virtual_leaves" => tree.virtual_leaves,
        "virtual_nodes" => tree.virtual_nodes,
        "levels" => tree.levels,
        "memory_indices" => [
            memory_index(tree, i)
            for i in 1:(2*t-1)
            if !isvirtual(tree, i)
        ],
    )
    # Save for Python tests.
end
```

Important: `ImplicitBVH.jl` uses 1-based indexing. CUDA code uses 0-based indexing. Adjust all comparisons by subtracting 1 from Julia indices where appropriate.

Validate `implicit_tree.cuh` for:

- `t = 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 100, 1000, 1023, 1024, 1025`
- real node count
- virtual leaf count
- leaf level
- virtual nodes per level
- `memory_index`
- `is_virtual`
- level real ranges
- parent/child relationships
- ancestor/descendant formulas

## Morton Encoding Tests

Test:

- 2D and 3D encode paths.
- Normalization into `[0, 1]`.
- Clamping at scene bounds.
- duplicate coordinates.
- known small integer Morton encodings.
- degenerate scene extents, if supported by implementation.

## BVH Build Validation

Current status: the Milestone 4 build work already added a focused
`tests/test_bvh_build.py` harness that covers the core validation checks below for small
2D and 3D point clouds, duplicate points, and degenerate extents. Milestone 5 should
expand this harness only where it adds new confidence before k-NN work, rather than
duplicating those existing checks.

Milestone 5 extends that same file rather than adding a second harness. The expanded
coverage should include powers of two and just-under/just-over powers of two, single and
two-point clouds, collinear 2D/3D inputs, planar 3D inputs, all-duplicate inputs, and
extreme aspect ratios.

Before implementing k-NN, validate:

- every real node has a finite valid AABB.
- root AABB matches the scene min/max.
- parent AABBs contain child AABBs.
- leaves are degenerate AABBs matching original points after `sorted_indices` lookup.
- virtual nodes are never addressed through `memory_index`; test helpers should require
  nodes to be real before reading their compact memory index.
- single-child internal nodes copy their real child AABB.
- single-child internal nodes are covered across multiple non-power-of-two sizes.
- sorted leaf positions map back to original point indices.
- `sorted_indices` is a permutation of the original point indices, preferably checked
  without moving the tensor to CPU unless the test needs host values.

## k-NN Validation

Compare against brute-force pairwise distance in PyTorch:

```python
def brute_force_knn(points, query_points, k):
    dists = torch.cdist(query_points, points)
    distances, indices = dists.topk(k, largest=False)
    return indices, distances
```

Test cases:

- Small clouds, around 100 points, where exact match is expected.
- Uniform random distributions in the unit square/cube.
- Points on a line.
- Points at the same location.
- A single outlier far away.
- Displaced queries, `query = point + small_dx`, with at least some neighbor changes.
- Self-query, `query_points == points`, where each point appears as its own nearest neighbor at distance 0.
- Tie behavior with identical points. Verify no crashes or invalid indices; exact tie order need not match brute force.

Output expectations:

- indices are original unsorted point indices.
- squared distances are sorted ascending.
- query wrappers that accept `source_points` should also provide neighbor positions with
  shape `(N_queries, k, D)` or the batched/ragged equivalent, gathered from source points
  by original or local per-sample indices.
- supported `k` values are exactly `4`, `8`, and `16`.

Stage 4 robustness policy:

- exact k-NN remains the required behavior across single-sample, fixed-size batched, and
  ragged paths;
- true equal-distance tie index order is not a stable public guarantee, so tie tests
  should validate exact distance sets, candidate-set membership, sorted distances, and
  valid local/original indices rather than deterministic ordering;
- non-ambiguous cases may still assert exact order when existing behavior depends on it;
- tests for large coordinate offsets with tiny displacements should prefer direct
  squared-distance broadcasting over `torch.cdist(...).square()` when the latter loses
  the displacement in float32.

## Invalid Numeric Inputs

Stage 5 accepts NaN and Inf inputs as undefined behavior for source points, query points,
displaced points, map source/target points, feature tensors, displaced-query values, and
map consumer values. The runtime does not scan inputs with `torch.isfinite`, and tests
should not assert a stable outcome for non-finite inputs. Treat finite-value hygiene as
caller responsibility: model and data-loading code must provide finite tensors before
calling `torchbvh`.

Keep tests focused on cheap structural validation such as rank, dtype, CUDA device,
contiguity, shape, handle lifecycle, supported `k`, selector arguments, and map-object
types. If a future stage adds finite-value guards, add focused `ValueError` tests only
after the new validation mode is accepted in `docs/decisions.md`.

## Robustness Regression Subsets

Future CI and benchmark-gate work should promote a curated subset of Stage 4 cases into
routine validation. Keep the fast subset small, deterministic, and focused on policy:
duplicates, all-duplicates, true ties, exact hits, collinear or planar neighborhoods,
tiny coordinate ranges, large coordinate offsets, mixed-scale axes, local-index
boundaries, sorted squared distances, and gradient boundaries.

Optional longer CUDA correctness jobs may use larger randomized cases that better match
the target training regime, including `B=16-32`, `N=10k-50k`, `k in {4, 8, 16}`, and
displaced-query volume from flattening `(B, N, H, D)` into `M = N * H` queries per
sample. These jobs should remain correctness-oriented unless they are explicitly part of
the Stage 8 benchmark gate.

## Python API Lifecycle Tests

Keep public wrapper tests focused in `tests/test_python_api.py`. These tests should verify
the ergonomic API and lifecycle behavior without duplicating the full brute-force k-NN
matrix from `tests/test_knn.py`.

Test:

- `build_bvh` returns a live mapping-like `BVHHandle`.
- `query_knn` works through that handle in both 2D and 3D.
- `query_knn` still accepts a legacy mapping payload where practical.
- unsupported `k` raises `ValueError` with `k must be 4, 8, or 16` on public wrapper
  surfaces.
- `destroy_bvh` marks a handle destroyed, releases tensor references, and causes later
  query/index access to raise a clear destroyed-handle error.
- destroying any handle type twice is idempotent, and `RaggedBVHHandle.destroy()`
  cascades to its inner per-sample handles.
- invalid handle types raise a clear `TypeError`.
- `SUPPORTED_K` and `SUPPORTED_DIMS` are importable from `torchbvh` and
  `torchbvh.ops`.

Native build wrappers may still surface `RuntimeError` for structural errors checked by
the C++/CUDA binding. Do not rewrite those tests to expect `ValueError` unless
`docs/decisions.md` changes the build-validation split.

## Class-Based BVH API Tests

Concrete coverage lives in `tests/test_bvh_class.py`.

Test:

- `BVH`, `BatchedBVH`, and `RaggedBVH` are importable from `torchbvh` and
  listed in `torchbvh.__all__`.
- Classes do not expose `__getitem__` or the `Mapping` protocol.
- `BVH(points)`: `.destroyed` is `False` before `destroy()`; `True` after. Double
  `destroy()` is idempotent. The context manager calls `destroy()` on `__exit__`.
  Calling `.knn()` after `destroy()` propagates the handle's `RuntimeError`.
- `BVH.knn(query_points, k)` returns results identical to `query_knn` on the same data.
- `BVH.knn(query_points, k, source_points=points)` returns results identical to `query_knn`
  with `source_points`, including neighbor positions.
- `BVH.interpolate(displaced_points, features, k)` returns results identical to
  `bvh_mls_interpolate`; default return is a plain `Tensor`; `return_grad=True` returns
  `(interpolated, field_gradient)` with correct shapes.
- `BatchedBVH`: same lifecycle and `.knn()` delegation tests for `(B, N, D)` inputs.
- `RaggedBVH`: lifecycle and `.knn()` delegation tests for packed ragged inputs.
- `RaggedBVH.interpolate()` raises `TypeError` with a clear message.

## Autograd Tests

For `bvh_mls_interpolate` and `bvh_mls_interpolate_batched`:

- interpolation should use local linear MLS, not inverse-distance weighting.
- forward output should match a simple PyTorch MLS reference using the same BVH-selected neighbors.
- the MLS reference test should not duplicate the full brute-force k-NN matrix; existing k-NN tests own neighbor-selection correctness.
- gradients should flow to `features`.
- gradients should flow to `displaced_points` through the differentiable MLS computation.
- gradients should be `None` for source `points` and `k`.
- BVH construction and discrete neighbor selection should remain non-differentiable.
- direct `query_knn*` optional neighbor-position gathers may propagate gradients to the
  explicit `source_points` tensor only; `BVHQuery` and `BatchedBVHQuery` must detach
  neighbor positions. This is an accepted asymmetry to document and test, not a runtime
  mismatch.
- include duplicate or zero-distance cases to exercise exact-source-point behavior.
- include collinear or planar neighborhoods and `k=4` in 3D to cover regularized, underdetermined MLS solves.
- verify that the default call returns a plain `Tensor` (not a tuple); verify that
  `return_grad=True` returns a two-tuple `(interpolated, field_gradient)` where
  `field_gradient` has shape `(N_queries, D, F)` and is usable as an auxiliary output.

## Native Batched Training Tests

The fixed-size batched path should be validated as a first-class API, not as a wrapper
around a Python loop. Tests may compare against a Python loop over the existing
single-sample APIs as an oracle, but the public batched calls must execute through batched
native bindings.

Initial test files should be split by behavior to keep failures readable:

- `tests/test_batched_python_api.py` for handle lifecycle, public validation, shape
  contracts, and invalid inputs.
- `tests/test_batched_knn.py` for exact batched k-NN correctness.
- `tests/test_batched_interpolate.py` for batched MLS forward and gradients.
Build/query tests:

- Cover `points: (B, N, D)` with `B > 1`, `D in {2, 3}`, and `N` values including powers
  of two and non-powers of two.
- Include small pathological clouds from the single-sample suite: duplicate points,
  collinear 2D/3D points, planar 3D points, degenerate extents, and non-power-of-two
  virtual-node boundaries.
- Verify `build_bvh_batched` returns a live `BatchedBVHHandle` with stacked payloads:
  `node_aabbs.shape == (B, Nr, 2*D)` and `sorted_indices.shape == (B, N)`.
- Verify each sample's `sorted_indices[b]` is a permutation of `0..N-1` and does not
  contain flattened cross-batch indices.
- Verify query outputs have shape `(B, M, k)` for indices/distances and `(B, M, k, D)` for
  neighbor positions when `source_points` is provided.
- Verify neighbor indices are local to each sample and match brute-force PyTorch per
  sample. Distances must be sorted ascending along the `k` dimension.
- Verify the batched output matches the single-sample loop oracle for representative
  cases, but do not implement the batched API itself as that loop.
- Cover `k in {4, 8, 16}` and reject all other `k` values with the same public message as
  the single-sample path.

Validation tests:

- Reject invalid handle types and destroyed batched handles clearly.
- Reject non-CUDA tensors, non-float32 point/query tensors, bad ranks, unsupported
  dimensions, mismatched batch sizes, mismatched `D`, and too few points for the
  requested `k`. Accept non-contiguous public inputs and compare them against
  contiguous equivalents.
- Reject single-sample handles passed to batched APIs and batched handles passed to
  single-sample APIs unless a wrapper explicitly documents support.
- For fixed-size batching, reject ragged inputs or offset-based arguments until the ragged
  milestone exists.

Batched MLS tests:

- Verify `bvh_mls_interpolate_batched(points, displaced_points, features, k)` returns a
  plain `Tensor` with shape `(B, M, C)` by default; verify that passing `return_grad=True`
  returns a two-tuple `(interpolated, field_gradient)` where `field_gradient.shape ==
  (B, M, D, C)`.
- Compare forward output against applying the existing single-sample MLS path per sample.
- Verify gradients flow to `features` and `displaced_points`.
- Verify gradients do not flow through source `points`, BVH construction, or discrete
  neighbor selection.
- Include exact-hit, duplicate-point, and underdetermined-neighborhood cases.

## Displaced-Query Tests

Concrete coverage lives in `tests/test_displaced_query.py` under the stable Stage 5 names.

Test:

- `query_displaced_knn(pos, q, k)` accepts `pos: (B, N, D)` and `q: (B, N, H, D)`,
  builds one BVH per sample, flattens only the query dimension, and returns local
  per-sample indices and sorted squared distances shaped `(B, N, H, k)`.
- optional neighbor positions have shape `(B, N, H, k, D)` and are detached.
- `gather_neighbor_values(values, indices)` gathers from matching heads only and returns
  `(B, N, H, k, Ch)`.
- `interpolate_displaced(..., reduction="weighted_mean")` returns `(B, N, H, Ch)`,
  handles exact hits by averaging zero-distance neighbor values, and rejects unsupported
  reductions with `ValueError`.
- gradients flow to `values` only; `pos`, `q`, integer indices, distances, and gathered
  neighbor positions remain non-differentiable.
- validation covers unsupported `k`, wrong shape, dtype, device, contiguity, and local
  source-index range.

## FPS Resolution Geometry Tests

FPS tests should keep the downsampling geometry contract separate from model policy:

- `fps` returns selected anchors, gathered anchor coordinates, fine-to-anchor assignment
  metadata, per-anchor radius/count metadata, Morton coarse ordering, and
  selection-order diagnostics;
- gradients should not flow through FPS anchor selection or assignment metadata;
- former BVH map builders/consumers are removed and should not appear in public API tests.
  builders/consumers remain deferred;
- Stage 6 should profile map materialization, optional relative-offset/mask tensors,
  repeated map construction, source-position gathers, and value gathers before adding
  ragged maps, native map handles, fused kernels, serialization, or differentiable map
  geometry;
- wrong map-object types should raise `TypeError`, while semantic validation failures
  should raise `ValueError`.

## Performance Benchmarks

`benchmarks/benchmark_knn.py` is the current benchmark entry point. During Stage 1, add
instrumentation there first unless a separate benchmark is explicitly justified. Keep
generated JSON artifacts out of source-controlled docs unless a task explicitly asks to
check in baseline results.

Stage 8 library-op optimization evidence uses the same entry point and generated JSON
under ignored artifact directories. The displaced-query report includes
`displaced_query_interpolate_ms` for the public `interpolate_displaced` helper in
addition to build, traversal, source-position gather, value gather, total path, peak GPU
memory, and CUDA allocator summaries. MLS evidence uses the existing
`gpu_mls_solve_ms` and `native_batched_mls_solve_ms` fields. Pre/post artifacts should
stay out of git unless a task explicitly asks to check them in.

Measure and compare against the CPU k-d tree baseline:

- BVH construction time vs. `scipy.spatial.cKDTree` construction.
- k-NN query time vs. `cKDTree.query`.
- total forward pass time with old vs. new implementation.
- feature and position gather time for any benchmark path that feeds learned consumers.
- GPU memory usage.

Primary benchmark regime:

- 10k-50k points per sample.
- batch size 16-32.
- single GPU.
- `k = 4, 8, 16`.
- 2D and 3D.

Expected target: significant speedup, greater than 10x on total spatial-query time, by removing CPU-GPU synchronization. Treat this as an expectation to confirm with measurement, not a guaranteed result.

Benchmark gate:

- Future gates should be milestone-specific: native fixed-size batching should be
  benchmarked before ragged batching or large single-cloud inference, and large-N work
  should not start until native batched training is proven.
- Optional benchmark features should stay flag-gated and lightweight by default.
- Add batched benchmark options before judging training throughput. The benchmark should
  report batched build, batched query, batched MLS, total batched path time, and peak GPU
  memory separately.
- Batched benchmarks should compare native batched execution against the Python loop over
  single-sample APIs to quantify Python-launch overhead and validate that the new path is
  materially useful for `B=16-32`, `N=10k-50k`.
- Benchmark defaults should remain lightweight, but at least one smoke command should
  exercise `B > 1` once the batched APIs exist.
- For single-sample cases, distinguish native k-NN traversal from the optional
  source-position gather. `query_knn(..., source_points=points)` returns gathered
  positions, but current benchmark reporting measures traversal as
  `gpu_bvh_query_traversal_ms`, gather as `gpu_position_gather_ms`, and their sum as
  `gpu_bvh_query_ms`.
- For internal optimization passes, collect pre/post evidence with focused commands such
  as:

  ```powershell
  python benchmarks/benchmark_knn.py --batch-size 16 --n 10000 --queries 400000 --dim 3 --k 8 --features 8 --displaced-query --displaced-query-heads 40 --displaced-query-channels 8 --iters 2 --warmup 1 --torch-cluster-baseline --output-json artifacts/stage8/2026-05-07/core_library_op_optimizations/post_displaced_b16_n10000_h40_ch8.json
  python benchmarks/benchmark_knn.py --batch-size 4 --n 1024 --queries 1024 --dim 3 --k 8 --features 32 --iters 5 --warmup 2 --output-json artifacts/stage8/2026-05-07/core_library_op_optimizations/post_mls_batched_b4_n1024_q1024_c32.json
  python benchmarks/benchmark_knn.py --n 10000 --queries 10000 --dim 3 --k 8 --features 160 --iters 5 --warmup 2 --output-json artifacts/stage8/2026-05-07/core_library_op_optimizations/post_mls_single_n10000_q10000_c160.json
  ```

## Ragged Batch Analysis

Ragged or variable-`N` point clouds are out of scope for the fixed-size batched milestone.
Before implementing ragged batching, define the packed tensor format, per-sample metadata,
output packing, and lifecycle behavior. Tests should cover different `N` and `M` values
per sample, empty samples only if explicitly supported, and no cross-sample neighbor
leakage.

Concrete Milestone 14 tests live in `tests/test_ragged_python_api.py` and
`tests/test_ragged_knn.py`.

Ragged build/query tests:

- Cover packed `points: (total_N, D)` and `query_points: (total_M, D)` with offsets of
  shape `(B + 1,)`, `D in {2, 3}`, and uneven per-sample `N` and `M`.
- Verify `build_bvh_ragged` returns a live `RaggedBVHHandle` with copied offset metadata,
  `batch_size`, `dim`, and per-sample point counts.
- Verify query outputs have packed shapes `(total_M, k)` for indices/distances and
  `(total_M, k, D)` for neighbor positions when `source_points` is provided.
- Verify neighbor indices are local to each sample and match looping over the existing
  single-sample `build_bvh`/`query_knn` APIs. Distances must be sorted ascending.
- Include duplicate points, 2D and 3D, `k in {4, 8, 16}`, and uneven batch sizes.

Ragged validation tests:

- Reject bad offsets: wrong rank, wrong dtype, wrong device, first offset not zero,
  final offset not matching the packed tensor length, and non-strictly-increasing
  offsets.
- Reject non-CUDA tensors, non-float32 point/query tensors, unsupported dimensions,
  mismatched batch sizes, mismatched dimensions, mixed devices, unsupported `k`, and
  source samples with fewer than `k` points. Accept non-contiguous public inputs and
  compare them against contiguous equivalents.
- Reject fixed-size batched and single-sample handles passed to ragged query APIs.
- Destroyed ragged handles should raise a clear destroyed-handle error.

## Large-N Single-Sample Inference Analysis

Before implementing a large-N single-sample branch, complete the native fixed-size batched
training path and run an analysis pass that keeps the public behavior fixed while
identifying the first actual bottleneck. The candidate threshold is `N > 128k`, but the
final threshold should be chosen from observed runtime, memory use, and failure behavior
rather than assumed from the roadmap.

The analysis pass should break down at least:

- scene-bound reduction.
- Morton code generation.
- sorting.
- BVH construction after sorting.
- k-NN traversal.
- optional neighbor position gather.
- Python/API overhead.
- temporary allocation and peak CUDA memory, where available.

Measure at increasing point counts, including at least one case below and above the
candidate threshold, while keeping `k in {4, 8, 16}` and both 2D and 3D represented where
practical. Report build, query, optional position-gather, total path time, and peak GPU
memory separately.

Benchmark reports for Milestone 15 should include `N`, `D`, `k`, number of queries,
build time, query time, optional position-gather time, total time, peak CUDA memory if
available, and whether the large-N branch was used. If profiling shows no separate branch
is justified, record that result instead of adding a speculative implementation.

Milestone 15 recorded that no separate large-N branch is currently justified. On the local
RTX 3500 Ada laptop GPU, profiling `N=262144`, `D=3`, `k=8`, and `4096` queries found
exact k-NN traversal to dominate the CUDA time. Scene-bound reduction, Morton coding,
sorting, bottom-up BVH construction, Python/API overhead, allocation, and the optional
position gather were small by comparison. Benchmarks should therefore report
`large_n_branch_used=False` for current single-sample cases.

Correctness checks for any large-N branch should preserve the same observable semantics:
exact k-NN distances matching brute force on smaller branch-forced cases, original point
indices, sorted neighbor distances, compact memory-index ordering for tree features, and
unchanged Python handle lifecycle. Large-N tests may force the branch at small `N` so the
branch can be covered without allocating inference-sized clouds in routine CI.

If a branch-force hook is added for tests, keep it private or debug-only, such as an
internal keyword, context manager, or environment variable. It should not appear in the
documented public API and should not change normal dispatch. Branch-forced tests should
also verify that fixed-size batched and ragged APIs are unchanged by Milestone 15.
Do not add branch-forced tests until a branch actually exists; without a branch, the
absence of a force hook is intentional.
