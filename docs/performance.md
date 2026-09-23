# Performance

`torchbvh` uses one maintained production path for each operation:

| Operation | Production implementation |
|---|---|
| Point BVH build | cooperative lower-level propagation with narrow Morton ordering |
| k-NN | cached child bounds, with ordered/spatial variants where query locality helps |
| MLS | indexed packed kernels; cooperative channel processing for wide features |
| FPS | exact bucketed or approximate bucket-queue sampling |
| Rays | cached segment traversal and general triangle traversal |

From a repository checkout, run the maintained benchmark entry points:

```bash
python benchmarks/benchmark_core.py --warmup 3 --iterations 10
python benchmarks/benchmark_conditional_mls.py --help
python benchmarks/benchmark_raytrace.py --help
```

`benchmark_core.py` records build, k-NN, MLS, exact/approximate FPS, and segment/triangle ray timings through the public API. Use identical arguments and hardware when comparing revisions.
Pass `--use-graph` to time FPS with CUDA graph capture enabled.

Benchmark after a warm-up, synchronize CUDA around timed regions, and report the PyTorch/CUDA versions, GPU, shapes, dtype, and command line. Compare output equivalence before comparing timings. Exact FPS should match the independent oracle; approximate FPS should be compared by assignment invariants and quality bounds rather than against a removed diagnostic kernel.

Performance reports and one-off candidate implementations are intentionally not kept in the package tree. Git history is the archive for completed optimization studies. Add a benchmark only when it is expected to remain useful for current production behavior.

Cleanup changes are acceptable only when retained routes preserve their outputs and show no material regression on representative build, k-NN, MLS, FPS, and ray workloads in a CUDA-enabled release environment.
