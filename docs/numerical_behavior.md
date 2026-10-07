# MLS numerical behavior

MLS uses float32 CUDA arithmetic. Neighbor selection and the per-channel local
linear solve retain the existing MLS definition. Eligible 0.3.2 calls share
geometry calculations across channels; see the [API reference](api_reference.md#pointgeometry)
for dispatch conditions and fallback behavior.

## Gradients and reproducibility

Feature gradients use float32 atomic additions. Their order can vary between
runs, including in 0.3.1. Wide-channel query gradients in the geometry-owner
kernels accumulate channel contributions in a different order from the legacy
cooperative kernels. Floating-point addition is not associative: when large
contributions nearly cancel, a small absolute change can cause a large relative
difference. Exact forward agreement therefore does not imply identical gradients
or training trajectories.

Strict legacy-gradient comparisons remain in the test suite with their original
tolerances and assertions. Some cancellation-heavy cases fail these comparisons
in 0.3.2. Comparisons of repeated legacy feature-gradient calculations can also
fail. The release does not guarantee bitwise reproducible feature gradients or
strict gradient parity with 0.3.1. This limitation also applies when using
explicit `PointGeometry` objects.

## Validation

The maintained contract tests check values and first-order derivatives against
an independent double-precision weighted linear solve on well-conditioned
2D/3D cases. They cover query-only, feature-only and combined gradients,
64 channels, spatial-derivative outputs, and actual production-kernel dispatch.
Additional tests cover exact hits, duplicates, degenerate geometry, inactive
NaNs, non-contiguous inputs, geometry mutation and lifetime, streams, and CUDA
graphs. Passing these cases does not establish accuracy for every ill-conditioned
neighborhood.

Run the contract suite and the strict legacy diagnostics separately:

```bash
python -m pytest -q -m "not legacy_gradient_parity"
python -m pytest -q -m legacy_gradient_parity
```

The second command can fail for the numerical differences above. No diagnostic
is excluded by default; `python -m pytest -q` runs both groups. See
[testing](testing.md) for CUDA build requirements.
