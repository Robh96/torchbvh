"""Compile the production extension on first use and reuse it across sessions.

Installation only packages sources. Compilation uses the running interpreter's
PyTorch and CUDA toolkit, never an independently resolved build dependency.
"""

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import threading

from ._version import __version__


_SOURCE_DIR = Path(__file__).resolve().parent / "csrc"
_SOURCES = (
    "bindings.cpp",
    "bvh_build.cu",
    "fps_sample.cu",
    "knn_query.cu",
    "mls_fused.cu",
    "mls_geometry.cu",
    "morton_sort.cu",
    "ray_query.cu",
)
_CXX_FLAGS = ["/O2"] if sys.platform == "win32" else ["-O2"]
_CUDA_FLAGS = ["-O3", "--use_fast_math", "-lineinfo"]
_module = None
_thread_lock = threading.Lock()


def _require_torch():
    """Keep PyTorch selection under the user's control, with a useful error."""
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        raise ModuleNotFoundError(
            "torchbvh requires an existing PyTorch >=2.0 installation. Install "
            "CUDA-enabled PyTorch for your system from https://pytorch.org/get-started/locally/. "
            "torchbvh does not install or replace PyTorch."
        ) from error
    if int(torch.__version__.split(".", 1)[0]) < 2:
        raise RuntimeError("torchbvh requires PyTorch >=2.0; your PyTorch was left unchanged.")
    return torch


def _file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hash():
    digest = hashlib.sha256()
    for path in sorted(_SOURCE_DIR.iterdir()):
        if path.suffix in {".cpp", ".cu", ".cuh"}:
            digest.update(path.name.encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _compiler_identity(executable, args):
    resolved = shutil.which(str(executable))
    if resolved is None:
        raise RuntimeError(
            f"torchbvh cannot find compiler {executable!s}. Install a CUDA toolkit "
            "with NVCC and a supported C++ compiler. On Windows, use the matching "
            "Visual Studio developer terminal. PyTorch was left unchanged."
        )
    try:
        result = subprocess.run(
            [resolved, *args], capture_output=True, text=True, errors="replace", timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"torchbvh could not identify compiler {resolved}.") from error
    # cl prints its version to stderr and returns nonzero with no input file.
    return {"path": str(Path(resolved).resolve()), "version": result.stdout + result.stderr}


def _build_identity(torch, cpp_extension):
    if not torch.version.cuda:
        raise RuntimeError(
            "torchbvh requires CUDA-enabled PyTorch, but the installed PyTorch is CPU-only. "
            "Install a CUDA-enabled build yourself; torchbvh leaves PyTorch unchanged."
        )
    cuda_home = cpp_extension.CUDA_HOME
    if cuda_home is None:
        raise RuntimeError(
            "torchbvh cannot find the CUDA toolkit. CUDA-enabled PyTorch supplies runtime "
            "libraries, but compiling torchbvh also requires NVCC. Install a compatible "
            "toolkit and set CUDA_HOME or CUDA_PATH to its root."
        )
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if not arch_list and not torch.cuda.is_available():
        raise RuntimeError(
            "torchbvh needs a visible CUDA GPU to select compilation targets, or an explicit "
            "TORCH_CUDA_ARCH_LIST for compilation without a visible GPU."
        )
    targets = arch_list or sorted({
        torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())
    })
    nvcc = _compiler_identity(Path(cuda_home) / "bin" / "nvcc", ["--version"])
    release = re.search(r"release (\d+)\.(\d+)", nvcc["version"])
    if release is None:
        raise RuntimeError("torchbvh could not determine the CUDA toolkit version from NVCC.")
    if release.group(1) != torch.version.cuda.split(".", 1)[0]:
        raise RuntimeError(
            f"torchbvh found CUDA toolkit {release.group(0)}, but PyTorch was built for "
            f"CUDA {torch.version.cuda}. Select a toolkit with the same CUDA major version. "
            "PyTorch was left unchanged."
        )
    return {
        "schema": 1,
        "torchbvh": __version__,
        "sources": _source_hash(),
        "python": sys.version,
        "abi": sysconfig.get_config_var("SOABI"),
        "platform": [platform.system(), platform.machine()],
        "torch": torch.__version__,
        "torch_path": str(Path(torch.__file__).resolve()),
        "torch_config": torch.__config__.show(),
        "torch_cuda": torch.version.cuda,
        "cuda_home": str(Path(cuda_home).resolve()),
        "nvcc": nvcc,
        "cxx": _compiler_identity(os.environ.get("CXX", "cl" if os.name == "nt" else "c++"),
                                  [] if os.name == "nt" else ["--version"]),
        "targets": targets,
        "flags": {"cxx": _CXX_FLAGS, "cuda": _CUDA_FLAGS},
        "environment": {key: os.environ.get(key) for key in (
            "CC", "CXX", "INCLUDE", "LIB", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS",
        )},
    }


def _cached_module(directory, name, identity):
    """Only load complete builds for exactly this source/environment identity."""
    try:
        receipt = json.loads((directory / "build.json").read_text(encoding="utf-8"))
        filename = receipt["binary"]
        if receipt["identity"] != identity or Path(filename).name != filename:
            return None
        binary = directory / filename
        if _file_hash(binary) != receipt["binary_sha256"]:
            # Otherwise Ninja could regard the damaged output as up to date.
            binary.unlink()
            return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    spec = importlib.util.spec_from_file_location(name, binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def _load_extension():
    global _module
    if _module is not None:
        return _module
    with _thread_lock:
        if _module is not None:
            return _module
        torch = _require_torch()
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Run a torchbvh operation before CUDA graph capture to load its native extension."
            )
        from filelock import FileLock
        from torch.utils import cpp_extension

        if not cpp_extension.is_ninja_available():
            raise RuntimeError("torchbvh needs Ninja to compile its extension: python -m pip install ninja")
        identity = _build_identity(torch, cpp_extension)
        # JSON normalizes capability tuples so the persisted identity compares exactly.
        encoded = json.dumps(identity, sort_keys=True)
        identity = json.loads(encoded)
        fingerprint = hashlib.sha256(encoded.encode()).hexdigest()[:32]
        name = f"_torchbvh_native_{fingerprint}"
        root = Path(os.environ.get("TORCH_EXTENSIONS_DIR") or cpp_extension.get_default_build_root())
        directory = root.expanduser().resolve() / "torchbvh" / fingerprint
        directory.mkdir(parents=True, exist_ok=True)
        # An OS-backed lock is released if a builder dies; another process can retry.
        with FileLock(str(directory / "build.lock"), timeout=1800):
            # PyTorch's inner file baton can survive a killed builder. Acquiring
            # our outer lock proves no other torchbvh builder owns this directory.
            (directory / "lock").unlink(missing_ok=True)
            module = _cached_module(directory, name, identity)
            if module is None:
                print("torchbvh: compiling native extension for your PyTorch/CUDA environment "
                      f"(cached in {directory})", file=sys.stderr)
                try:
                    module = cpp_extension.load(
                        name=name,
                        sources=[str(_SOURCE_DIR / source) for source in _SOURCES],
                        extra_cflags=_CXX_FLAGS,
                        extra_cuda_cflags=_CUDA_FLAGS,
                        extra_include_paths=[str(_SOURCE_DIR)],
                        build_directory=str(directory),
                        verbose=os.environ.get("TORCHBVH_BUILD_VERBOSE") == "1",
                        with_cuda=True,
                    )
                except (OSError, RuntimeError) as error:
                    raise RuntimeError(
                        "torchbvh native compilation failed. Check that NVCC and the host compiler "
                        "support your installed PyTorch/CUDA version. Set TORCHBVH_BUILD_VERBOSE=1 "
                        "for compiler output or MAX_JOBS to limit build parallelism. "
                        "PyTorch was left unchanged."
                    ) from error
                binary = Path(module.__file__)
                receipt = {"identity": identity, "binary": binary.name,
                           "binary_sha256": _file_hash(binary)}
                temporary = directory / "build.json.tmp"
                temporary.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
                temporary.replace(directory / "build.json")
            _module = module
        return _module


class _NativeExtension:
    """Resolve each native function once, without adding a wrapper to kernel calls."""

    def __getattr__(self, name):
        if name.startswith("__") and name != "__file__":
            raise AttributeError(name)
        value = getattr(_load_extension(), name)
        setattr(self, name, value)
        return value


_C = _NativeExtension()
