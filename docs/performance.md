# Performance

`torchbvh` uses one maintained production path for each operation:

| Operation | Production implementation |
|---|---|
| Point BVH build | cooperative lower-level propagation with narrow Morton ordering |
| k-NN | cached child bounds, with ordered/spatial variants where query locality helps |
| MLS | geometry-owner kernels for eligible K=4 calls; packed/cooperative fallback otherwise |
| FPS | exact bucketed or approximate bucket-queue sampling |
| Rays | cached segment and triangle traversal with fused first-order backward |

From a repository checkout, run the maintained benchmark entry points:

```bash
python benchmarks/benchmark_core.py --warmup 3 --iterations 10
python benchmarks/benchmark_conditional_mls.py --help
python benchmarks/benchmark_raytrace.py --help
python benchmarks/benchmark_raytrace.py --primitive-type triangle --compare-general
python benchmarks/benchmark_raytrace.py --primitive-type triangle --scene mesh --compare-general
python benchmarks/benchmark_raytrace.py --primitive-type triangle --scene miss --compare-general
```

`benchmark_core.py` records build, k-NN, MLS, exact/approximate FPS, and segment/triangle ray timings through the public API. `benchmark_raytrace.py --compare-general` checks cached triangle outputs against the retained general traversal and reports same-process forward and forward-plus-backward timings. Use identical arguments and hardware when comparing revisions.
Pass `--use-graph` to time FPS with CUDA graph capture enabled.

The [third-party benchmark notebook](https://github.com/Robh96/torchbvh/blob/main/examples/third_party_benchmarks.ipynb) compares one-shot k-NN, FPS, and interpolation against SciPy, torch-cluster, CuPy, fpsample, torch-fpsample, and PyTorch grid sampling. It displays the retained/refreshed dataset by default. Explicitly enable its full-sweep option to collect all methods again. It reports output quality alongside latency because approximate FPS and grid interpolation have different semantics.

The 0.3.2 figure refresh runs only the 40 torchbvh k-NN/MLS rows;
all 160 other rows remain unchanged. See the
[benchmark guide](https://github.com/Robh96/torchbvh/blob/main/benchmarks/README.md)
for commands, dates and measurement provenance, and
[numerical behavior](numerical_behavior.md) for gradient limitations.
The figure measures ordinary one-shot MLS; it does not measure conditional model training.

Benchmark after a warm-up, synchronize CUDA around timed regions, and report the PyTorch/CUDA versions, GPU, shapes, dtype, and command line. Compare output equivalence before comparing timings. Exact FPS should match the independent oracle; approximate FPS should be compared by assignment invariants and quality bounds rather than against a removed diagnostic kernel.

The benchmark directory contains maintained public-API comparisons and the
datasets needed to reproduce the performance figure. Add a benchmark only when
it is expected to remain useful for current production behavior.

Cleanup changes are acceptable only when retained routes preserve their outputs and show no material regression on representative build, k-NN, MLS, FPS, and ray workloads in a CUDA-enabled release environment.
