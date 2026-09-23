# Testing

The test suite requires a CUDA-enabled PyTorch installation, a supported NVIDIA GPU, NVCC, and a host compiler supported by your CUDA toolkit. Check the active interpreter before building:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

On Windows with CUDA 12.6, use a Visual Studio 2022 developer shell; Visual Studio 2026's compiler is rejected by this toolkit. Rebuild against the active PyTorch installation, then run the suite:

```bash
python setup.py build_ext --inplace --force
python -m pytest -q
```

The maintained suite covers the public workflows in two and three dimensions, with `k` values 4, 8, and 16 where applicable.
It includes single, fixed-batch, ragged, multihead, conditional, displaced-query, FPS, segment-ray, and triangle-ray cases.
Lifecycle, non-contiguous input, duplicate geometry, inactive-NaN routing, first-order gradient, and CUDA stream/graph behavior are also exercised.

Run a focused area while iterating:

```bash
python -m pytest -q tests/test_bvh_build.py tests/test_knn.py
python -m pytest -q tests/test_mls_interpolate.py tests/test_conditional_mls.py
python -m pytest -q tests/test_fps.py tests/test_raytrace.py
python -m pytest -q tests/test_python_api.py tests/test_readme_examples.py
```

Exact bucketed FPS is checked against an independent small-input oracle. Approximate FPS is checked through assignment invariants and quality bounds; the removed full-scan implementation is not used as production reference code.

Before release, install `build` and `twine` into a clean CUDA-enabled build environment with its intended PyTorch version. PyTorch selects the compute capabilities of visible GPUs by default.
To build for another GPU, set `TORCH_CUDA_ARCH_LIST` before compilation (for example, `TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9+PTX"` in Bash, or `$env:TORCH_CUDA_ARCH_LIST = "8.0 8.6 8.9+PTX"` in PowerShell, with a compatible CUDA toolkit). Set it explicitly if no GPU is visible in the build environment.
Disable PEP 517 build isolation so the extension builds against that PyTorch rather than an independently resolved build-time copy:

```bash
python -m build --sdist --no-isolation
python -m twine check dist/torchbvh-*.tar.gz
```

Inspect the sdist for all `csrc` sources, with no research files or generated artifacts. In a fresh CUDA-enabled environment on another GPU machine if possible, install the built sdist:

```bash
python -m pip install --upgrade setuptools wheel
python -m pip install --no-build-isolation dist/torchbvh-0.3.1.tar.gz
python -m pip check
```

Then run an import and the public examples. The release does not publish a prebuilt wheel: a wheel compiled against one PyTorch/CUDA/GPU combination is not generally safe to offer to other combinations.

Run the maintained core benchmark on the same GPU and input sizes for the baseline and release candidate. Compare exact outputs and approximate-FPS quality before accepting timings. Finally, build documentation with strict link
checking:

```bash
mkdocs build --strict
```

CPU-only environments can still run syntax and static-reference checks, but cannot import the compiled extension or validate CUDA behavior.
