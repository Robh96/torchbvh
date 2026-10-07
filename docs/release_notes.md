# Release notes

## 0.3.2

Eligible ordinary and conditional MLS calls automatically use compiled two-lane
geometry-owner kernels over native feature banks, with 2D/3D saved-state layouts.
K=4 and channels 4/8/16/32/64 are eligible at >=4096 total queries; other cases
retain existing fallbacks. Geometry reuse stays explicit and forward-scoped.
`experimental_fast` remains accepted for compatibility.

The strict legacy gradient-parity diagnostics remain unresolved: feature scatter
uses nondeterministic FP32 atomics, and wide-channel query reductions have a
different summation order. These numerical differences are accepted for 0.3.2;
the assertions remain unchanged. See [numerical behavior](numerical_behavior.md)
for the mathematical reference checks and limitations.

Only torchbvh's 40 exact k-NN/MLS figure rows were refreshed. The other 160 rows,
including FPS and grid_sample, retain their original measurements.
The distribution remains source-only, built against the user's PyTorch/CUDA
environment.
