# Examples

The repository includes small notebook-style examples under `examples/`. They use
tiny tensors and simple PyTorch losses to show where each geometry primitive fits
in an unstructured point-cloud training step.

Each notebook is CUDA-gated. If CUDA is unavailable, the notebook prints a clear
skip message instead of running the CUDA cells.

## Notebooks

| Notebook | Demonstrates |
| --- | --- |
| [`basic_bvh_knn.ipynb`](https://github.com/Robh96/torchbvh/blob/main/examples/basic_bvh_knn.ipynb) | local point-neighborhood feature aggregation with `BVH(points)` and `bvh.knn(...)` |
| [`mls_interpolation.ipynb`](https://github.com/Robh96/torchbvh/blob/main/examples/mls_interpolation.ipynb) | MLS feature sampling at learned offset positions, including `return_grad=True` field gradients |
| [`batched_displaced_query.ipynb`](https://github.com/Robh96/torchbvh/blob/main/examples/batched_displaced_query.ipynb) | a tiny multihead displaced-query block with `query_displaced_knn`, `gather_neighbor_values`, and `interpolate_displaced` |
| [`fps_downsampling_geometry.ipynb`](https://github.com/Robh96/torchbvh/blob/main/examples/fps_downsampling_geometry.ipynb) | point-cloud downsampling with `fps(...)` |