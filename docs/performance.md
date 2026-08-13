# Performance

`torchbvh` is optimized first for training-oriented point-cloud workloads:
10k-50k points per sample, fixed-size batches of 16-32 samples, and one GPU.
Large single-cloud inference is supported through the same public APIs, but it is secondary to the batched training path.

This page explains benchmark output. It is not a new performance guarantee and does not define pass/fail thresholds.

## Maintained Benchmark Entry Points

Use the benchmark scripts as measurement tools on your own hardware:

| Entry point | What it measures |
| --- | --- |
| `benchmarks/benchmark_knn.py` | single-sample, fixed-size batched, ragged, and displaced-query k-NN paths; optional MLS timing; optional CPU or third-party baselines when their packages are installed |
| `benchmarks/benchmark_fps.py` | public FPS geometry plus a standalone FPS-warp interpolation proxy at small, 10k-scale, and 50k-scale cases |
| `benchmarks/benchmark_raytrace.py` | primitive-BVH build, reused closest-hit traversal, and one-shot ray tracing |
| `benchmarks/benchmark_conditional_mls.py` | two-MLS-plus-`where` baseline versus one-query-per-route conditional MLS, including forward, forward/backward, and peak allocation |
| `benchmarks/benchmark_flowers_grid_sample.py` | diagnostic comparison between a grid-sample Flower block and the accepted packed-head BVH MLS proxy |
| `tools/training_step_proxy.py` | small CUDA diagnostic that composes public displaced-query, MLS, and FPS APIs in one proxy step |

Ray tracing should be measured as separate build, reused traversal, and one-shot
build-plus-traversal timings. Its expected advantage appears once pruning avoids
enough primitive tests to repay Morton sorting and tree traversal; the crossover
depends on primitive distribution, ray coherence, `B`, `F`, and ray count. For
small fields, a dense intersection can remain faster despite worse scaling.

Conditional MLS timings compare equivalent user-visible results. The baseline
builds/searches both point sets and solves MLS twice for all `R=M*H` queries;
the conditional path still builds both BVHs but sorts and traverses exactly one
route and performs one MLS solve per query. Its benefit therefore grows when
query traversal and MLS dominate the two unavoidable builds. Read
`baseline_forward_ms` and `conditional_forward_ms` as inference-style timings,
the `*_forward_backward_ms` fields as training timings, and `*_peak_mib` as
incremental peak allocated memory for either forward or forward/backward. The maintained script defaults to
`B=16`, `N_field=M=16000`, `N_boundary=1600`, and `H=40`; reduce `--queries`
or `--batch` for a smoke run on smaller GPUs.

Local production-shape reference (RTX 3500 Ada, CUDA float32, `k=4`, 20% true
routes, one channel per head; two timed forward iterations and one
forward/backward iteration):

| D | Baseline forward | Routed forward | Speedup | Baseline fwd+bwd | Routed fwd+bwd | Speedup | Training peak baseline / routed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 211.7 ms | 113.0 ms | 1.87x | 804.7 ms | 435.6 ms | 1.85x | 3014 / 1608 MiB |
| 3 | 390.9 ms | 220.0 ms | 1.78x | 1176.2 ms | 652.6 ms | 1.80x | 4031 / 2155 MiB |

Inference-only peak allocation was about 20-22 MiB higher for the routed path
because it concatenates the two source banks, while the sequential baseline can
release branch-local search temporaries. During forward/backward, avoiding the
second saved MLS graph reduced peak allocation by about 1.4 GiB in 2-D and
1.8 GiB in 3-D. These are reference measurements, not release thresholds.

Optional third-party comparison notebooks and optional baseline flags are useful for experiments, but they are outside routine validation.

## Benchmark Fields

Benchmark rows usually report a mean over timed iterations, with `min`, `max`, and `stdev` in JSON summaries when available. Read fields by the operation they isolate:

| Field family | Meaning |
| --- | --- |
| build time | BVH construction for the measured source geometry, such as `gpu_bvh_build_ms`, `native_batched_build_ms`, `source_bvh_build_ms`, or `ragged_build_ms` |
| traversal/query time | native exact k-NN traversal, such as `gpu_bvh_query_traversal_ms`, `native_batched_query_traversal_ms`, `query_traversal_ms`, or `displaced_query_traversal_ms` |
| gather time | tensor gathers after discrete indices are known, such as source-position gathers, neighbor-feature gathers, seed gathers, and displaced-query value gathers |
| MLS time | local MLS interpolation work, such as `gpu_mls_solve_ms`, `native_batched_mls_solve_ms`, `mls_fused_chunk_forward_ms`, `mls_total_ms`, or `bvh_mls_interpolate_ms` |
| FPS geometry time | farthest-point anchor selection and assignment metadata, usually `fps_geometry_ms` |
| total path time | the benchmark's composed package path for that row, such as `gpu_total_bvh_path_ms`, `native_batched_total_path_ms`, `displaced_query_total_path_ms`, or `total_forward_ms` |
| peak allocated memory | maximum live bytes reported by PyTorch's CUDA allocator for timed iterations, for example `peak_gpu_memory_bytes` or `cuda_memory_allocated_peak_bytes` |
| peak reserved memory | maximum cached/reserved bytes held by PyTorch's allocator, for example `peak_reserved_memory_bytes` or `cuda_memory_reserved_peak_bytes` |

Build, traversal, gather, and MLS fields are intentionally separate. For example, `query_knn(..., source_points=points)` returns gathered neighbor positions, but benchmarks may report native traversal and the later position gather as separate
rows before summing them into a query or total path field.

## Smoke Versus Production Evidence

Smoke benchmarks check that a maintained path runs and that its row fields still exist. They may use tiny inputs, few iterations, optional skips, and whatever GPU is available locally.

Production evidence needs controlled variance: fixed commands, recorded hardware, driver, CUDA, PyTorch, build architecture, warmup and iteration counts, clean git metadata, retained baseline artifacts, and an accepted noise policy. The project has not accepted benchmark pass/fail thresholds or variance rules for public performance claims. Do not treat a smoke row,
README example number, or one-off laptop timing as a release gate.

Timings are hardware dependent. GPU model, driver, CUDA toolkit, PyTorch version, compiled architecture, input distribution, `B`, `N`, query count, `D`, `k`, feature channels, and allocator state can all change results. Compare variants on the same machine and build when deciding whether a path is faster for your case.

## FPS Interpretation

`fps(...)` currently exposes exact and approximate public modes:

- `mode="exact_bucketed"` is the default exact path.
- `mode="exact_full_scan"` is an exact full-scan fallback.
- `mode="approx_bucketed"` is an opt-in approximate anchor-selection path.

FPS rows should be read as geometry timing plus metadata production, not learned pooling quality. `fps_geometry_ms` measures anchor selection and assignment metadata. Rows such as seed gather, offset construction, interpolation, mixer, forward, backward, and train-step time belong to the benchmark proxy around FPS, not to the `fps(...)` API itself.

The approximate mode may improve throughput by relaxing the exact global anchor order. Its assignments are exact for the anchors it selected, but it is not expected to match the exact serial FPS sequence.

## Large Single-Cloud Inference

No separate large-`N` inference branch is currently promised. Current profiling found exact k-NN traversal to be the dominant large single-cloud cost on the local development GPU; build, sort, gather, allocation, and Python overhead were smaller by comparison. A new branch would need measured evidence that it improves a real bottleneck while preserving the same public exact k-NN behavior.

For now, use the same `BVH`, `build_bvh`, and `query_knn` surfaces and measure on your own hardware. Current benchmark metadata should report `large_n_branch_used=False` for the maintained single-sample path.
