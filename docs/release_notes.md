# Release notes

## 0.3.4

Correct MLS query gradients to include the adaptive lower-median Gaussian
bandwidth derivative in geometry-owner, packed/routed, and cooperative CUDA
backward kernels. Earlier releases through 0.3.3 froze bandwidth during backward.
Forward values, fitted linear-coefficient outputs, feature-gradient formulas,
source-coordinate detachment, and public signatures are unchanged. Query gradients
and training trajectories can change. No compatibility flag is required.

Clarify that `return_grad=True` returns fitted linear coefficients rather than
the autograd query derivative of adaptive MLS. Neighbor identities/order remain
discrete; the bandwidth derivative is zero at or below its floor. See
[numerical behavior](numerical_behavior.md) for the gradient contract and tests.

## 0.3.3

Plain `pip install torchbvh` no longer resolves a separate build-time PyTorch.
PyTorch is an explicitly managed prerequisite: torchbvh neither installs nor
replaces it. Installation packages sources, and importing the package does not
compile the native extension.

The first operation compiles the existing production CUDA kernels against the
running environment. Subsequent sessions load the cached binary. Cache identities
include the source, Python/PyTorch build, toolkit, host compiler, compiler flags,
and GPU targets. Builds are serialized across threads and processes; incomplete
builds can be retried. CUDA graph capture requires an operation first to warm up
the extension. See [Testing and builds](testing.md).

The algorithms, compiler optimization flags, public operation signatures, and
documented numerical limitations are unchanged. The performance figure retains
its explicitly labeled 0.3.2 measurements.

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
