# Public benchmarks

Run these tools from a repository checkout with torchbvh built for the active
CUDA-enabled PyTorch environment. The source distribution contains the library,
documentation and tests; these comparison tools are available in the repository.

| Tool | Purpose |
|---|---|
| `benchmark_core.py` | Build, exact k-NN, MLS, FPS and ray tracing through public APIs |
| `benchmark_conditional_mls.py` | Conditional MLS versus evaluating both branches; forward/backward and memory |
| `benchmark_raytrace.py` | Segment/triangle workloads, including comparison with general traversal |
| `third_party.py` | Shared fixtures, quality checks and third-party timing protocol |
| `refresh_torchbvh.py` | Refresh only torchbvh exact k-NN and MLS in an existing 200-row dataset |
| `render_performance_story.py` | Render SVG/PNG from measurements without running benchmarks |

```bash
python benchmarks/benchmark_core.py --warmup 3 --iterations 10
python benchmarks/benchmark_conditional_mls.py --help
python benchmarks/benchmark_raytrace.py --help
```

## Performance figure

The figure uses 20 workloads per method: B=1/16, D=2/3 and 10k–50k sources
and queries. Exact k-NN uses K=4. MLS uses K=4 and 64 channels. FPS selects
25% of sources. Inputs are float32 and prestaged in native CPU/GPU memory;
generation and transfers are excluded. A timed call includes its own tree
construction or setup. Timing uses one warmup and the median of three
synchronized calls. Quality checks run outside timing.

Interpolation uses a regular lattice so grid_sample has favorable inputs. MLS
and bilinear/trilinear interpolation use different weights; their analytic-field
errors are reported alongside latency. These are ordinary one-shot MLS timings,
not conditional training measurements. CPU/GPU methods and FPS return types
also differ; see the [notebook](../examples/third_party_benchmarks.ipynb).

The immutable original measurements are in
[`third_party_0.3.1_2026-09-23.csv`](results/third_party_0.3.1_2026-09-23.csv).
The 0.3.2 dataset is
[`third_party_0.3.2_2026-10-07.csv`](results/third_party_0.3.2_2026-10-07.csv),
with [measurement provenance](results/third_party_0.3.2_2026-10-07.json).
Only 40 torchbvh exact k-NN/MLS rows were refreshed on 2026-10-07. All 160
competitor, grid_sample and FPS rows retain their exact original CSV text and
2026-09-23 measurements. The refreshed README k-NN average is the arithmetic
mean of the 20 per-workload torch_cluster/torchbvh ratios (25.3x).
Approximate FPS remains 9.2x faster than fpsample CPU across the same 20 workloads.

Render the checked-in data without CUDA or third-party benchmark packages:

```bash
python -m pip install matplotlib pandas numpy
python benchmarks/render_performance_story.py benchmarks/results/third_party_0.3.2_2026-10-07.csv docs/assets/performance_story_032
```

To collect a new selective refresh, install SciPy, CuPy, torch-cluster, fpsample
and torch-fpsample versions compatible with your PyTorch/CUDA environment, then:

```bash
python -m benchmarks.refresh_torchbvh benchmarks/results/third_party_0.3.1_2026-09-23.csv artifacts/third_party_new.csv
```

The refresh preserves the input CSV and checks that the other 160 rows remain
unchanged. Keep its JSON sidecar with the output. The notebook displays the
checked-in 0.3.2 data by default; a full sweep requires explicitly setting
`run_full_sweep = True`. Use identical hardware and protocol when comparing
revisions; these measurements do not predict performance on every GPU.
