# Testing and builds

## Installation and prerequisites

Install CUDA-enabled PyTorch >=2.0 for your system, a compatible CUDA toolkit
including NVCC, and a host C++ compiler supported by that toolkit. Then run:

```bash
python -m pip install torchbvh
```

Starting with 0.3.3, PyTorch is an explicitly managed prerequisite. torchbvh never
installs or replaces it. Installation packages the CUDA sources; neither package
installation nor `import torchbvh` compiles the extension. The first operation
compiles against the running interpreter's PyTorch and local toolkit.
Later operations and Python sessions reuse the cached binary.

Check the active interpreter before your first operation:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

CUDA-enabled PyTorch provides runtime libraries, but does not supply NVCC. Set
`CUDA_HOME` or `CUDA_PATH` if the local toolkit cannot be detected. On Windows,
run operations from a Visual Studio developer terminal compatible with NVCC.
For example, CUDA 12.6 supports Visual Studio 2022; Visual Studio 2026's compiler
is rejected by that toolkit.

## First compilation and cache reuse

The first operation reports the cache directory and may take several minutes:

```python
import torch
import torchbvh

points = torch.rand(32, 3, device="cuda")
with torchbvh.BVH(points) as bvh:
    indices, distances = bvh.knn(points[:4], k=4)
```

The cached extension contains torchbvh's C++/CUDA kernels. PyTorch itself is
never compiled. Native functions are resolved once and then called directly.
Warm up with an operation before CUDA graph capture.

The cache identity includes native sources and headers, torchbvh version, Python
ABI, the installed PyTorch build and location, CUDA toolkit, host compiler,
optimization flags, and GPU targets. A new Python session selects a separate
cache when these change. Concurrent builders share an OS-backed lock, and a
successful build records the binary hash before it becomes reusable.
An already running process continues using the extension it loaded; restart
Python after changing build settings or replacing PyTorch.

The default location is PyTorch's user cache, under `torchbvh/<fingerprint>`.
Set `TORCH_EXTENSIONS_DIR` to choose a writable persistent cache directory.
Set `TORCHBVH_BUILD_VERBOSE=1` for compiler output or `MAX_JOBS` to limit build
parallelism. For example, in PowerShell:

```powershell
$env:TORCH_EXTENSIONS_DIR = "C:\torch-extension-cache"
$env:MAX_JOBS = "2"
$env:TORCHBVH_BUILD_VERBOSE = "1"
```

PyTorch targets visible GPUs by default. Set `TORCH_CUDA_ARCH_LIST` before the
first operation to target a broader set, for example `"8.0 8.6 8.9+PTX"` with a
supporting toolkit. Set it explicitly to compile without a visible GPU.
If the cache is removed, the next operation recompiles.

## Source development and regression tests

Install a checkout with the same ordinary workflow, or use editable mode:

```bash
python -m pip install -e .
python -m pytest -q
```

Native source changes select a new cache in the next Python process; no in-place
`setup.py build_ext` step is needed. The loader does not use old in-place `_C`
binaries. The suite requires CUDA for GPU operations and compiles on its first
native call if needed.

The maintained suite covers single, fixed-batch, ragged, multihead, conditional,
displaced-query, FPS, segment-ray, and triangle-ray cases in 2D/3D. Lifecycle,
non-contiguous input, duplicates, inactive NaNs, gradients, and CUDA streams and
graphs are exercised. Loader tests separately cover import behavior, environment
checks, cache invalidation, locking, and retry behavior.

```bash
python -m pytest -q tests/test_native_loader.py
python -m pytest -q tests/test_bvh_build.py tests/test_knn.py
python -m pytest -q tests/test_mls_interpolate.py tests/test_conditional_mls.py
python -m pytest -q tests/test_fps.py tests/test_raytrace.py
```

The blocking contract suite and retained strict legacy parity diagnostics run
separately:

```bash
python -m pytest -q -m "not legacy_gradient_parity"
python -m pytest -q -m legacy_gradient_parity
```

The second command retains its original strict assertions and can fail because
of the [documented FP32 accumulation differences](numerical_behavior.md).
No marker is excluded by default. Any unrelated correctness failure remains a
release blocker.

## Distribution checks

Release artifacts remain source distributions. Building an sdist does not need
PyTorch, CUDA, or a host compiler:

```bash
python -m build --sdist
python -m twine check dist/torchbvh-*.tar.gz
```

Inspect the sdist and the installation-generated Python wheel for every native
source/header, with no compiled workstation binary or research artifact. In a
fresh environment, install PyTorch independently, then install the exact sdist
using default build isolation:

```bash
python -m pip install dist/torchbvh-0.3.4.tar.gz
python -m pip check
```

Verify PyTorch's version and files remain unchanged. Use an empty persistent
cache to check that import does not compile, the first operation does, and a
second Python process loads the same binary without invoking the compiler.
Run public examples, the contract suite, and sequential paired performance
checks. Build documentation with `mkdocs build --strict`.

CPU-only PyTorch can import torchbvh without compiling; CUDA operations report
the missing CUDA-enabled PyTorch prerequisite. Without PyTorch, installation
still succeeds and import reports how to install the prerequisite.

Publish only after the local diff and tested sdist are reviewed. Verify GitHub
first, then upload that same source artifact through Twine.
