"""Loader contracts without compiling or changing the user's PyTorch."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import pytest
import torch
from torch.utils import cpp_extension

from torchbvh import _extension as extension


@pytest.fixture
def fake_build(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(extension, "_module", None)
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(cpp_extension, "is_ninja_available", lambda: True)
    monkeypatch.setattr(extension, "_build_identity", lambda *_: {"source": "current", "torch": "existing"})

    def compile_stub(**kwargs):
        calls.append(kwargs)
        directory = Path(kwargs["build_directory"])
        binary = directory / (kwargs["name"] + ".pyd")
        binary.write_bytes(b"native test binary")
        return SimpleNamespace(__file__=str(binary), operation=lambda: "native")

    monkeypatch.setattr(cpp_extension, "load", compile_stub)
    return calls


def test_install_metadata_never_imports_or_depends_on_torch(monkeypatch):
    import setuptools

    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    assert not any(requirement.startswith("torch") for requirement in config["build-system"]["requires"])
    metadata = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: metadata.update(kwargs))
    monkeypatch.setitem(sys.modules, "torch", None)
    runpy.run_path(str(root / "setup.py"))
    assert metadata["version"] == extension.__version__
    assert not any(requirement.startswith("torch") for requirement in metadata["install_requires"])
    assert "ext_modules" not in metadata
    assert {"csrc/*.cpp", "csrc/*.cu", "csrc/*.cuh"} <= set(metadata["package_data"]["torchbvh"])


def test_import_in_new_process_does_not_load_or_compile(tmp_path):
    code = """
import torch.utils.cpp_extension as cpp
def forbidden(*args, **kwargs):
    raise AssertionError('import tried to compile')
cpp.load = forbidden
import torchbvh
from torchbvh import _extension
assert _extension._module is None
assert not _extension._C.__dict__
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_missing_torch_has_actionable_error():
    code = """
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'torch':
            raise ModuleNotFoundError('blocked torch', name='torch')
sys.meta_path.insert(0, NoTorch())
try:
    import torchbvh
except ModuleNotFoundError as error:
    assert 'does not install or replace PyTorch' in str(error), str(error)
else:
    raise AssertionError('missing prerequisite accepted')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_native_functions_are_resolved_once(fake_build):
    proxy = extension._NativeExtension()
    assert proxy.operation() == "native"
    assert proxy.operation() == "native"
    assert len(fake_build) == 1
    assert proxy.operation is extension._module.operation
    assert fake_build[0]["extra_cflags"] == extension._CXX_FLAGS
    assert fake_build[0]["extra_cuda_cflags"] == extension._CUDA_FLAGS


def test_concurrent_threads_compile_once(fake_build):
    with ThreadPoolExecutor(max_workers=8) as pool:
        modules = list(pool.map(lambda _: extension._load_extension(), range(16)))
    assert all(module is modules[0] for module in modules)
    assert len(fake_build) == 1


def test_complete_cache_load_skips_build(fake_build, monkeypatch):
    expected = extension._load_extension()
    monkeypatch.setattr(extension, "_module", None)
    loader = SimpleNamespace(exec_module=lambda module: None)
    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *_: SimpleNamespace(loader=loader))
    monkeypatch.setattr(importlib.util, "module_from_spec", lambda *_: expected)
    assert extension._load_extension() is expected
    assert len(fake_build) == 1


@pytest.mark.parametrize("change", ["source", "torch"])
def test_identity_change_selects_separate_cache(fake_build, monkeypatch, change):
    extension._load_extension()
    monkeypatch.setattr(extension, "_module", None)
    identity = {"source": "current", "torch": "existing", change: "changed"}
    monkeypatch.setattr(extension, "_build_identity", lambda *_: identity)
    extension._load_extension()
    assert len(fake_build) == 2
    assert fake_build[0]["build_directory"] != fake_build[1]["build_directory"]


def test_build_failure_can_be_retried(fake_build, monkeypatch):
    original = cpp_extension.load
    monkeypatch.setattr(cpp_extension, "load", lambda **_: (_ for _ in ()).throw(RuntimeError("bad compiler")))
    with pytest.raises(RuntimeError, match="PyTorch was left unchanged"):
        extension._load_extension()
    assert extension._module is None
    monkeypatch.setattr(cpp_extension, "load", original)
    assert extension._load_extension().operation() == "native"
    assert len(fake_build) == 1


def test_interrupted_torch_build_lock_is_removed(fake_build, monkeypatch, tmp_path):
    original = cpp_extension.load

    def interrupted(**kwargs):
        (Path(kwargs["build_directory"]) / "lock").touch()
        raise RuntimeError("interrupted build")

    monkeypatch.setattr(cpp_extension, "load", interrupted)
    with pytest.raises(RuntimeError):
        extension._load_extension()

    def retry(**kwargs):
        assert not (Path(kwargs["build_directory"]) / "lock").exists()
        return original(**kwargs)

    monkeypatch.setattr(cpp_extension, "load", retry)
    assert extension._load_extension().operation() == "native"


def test_corrupted_binary_is_removed_before_rebuild(fake_build, monkeypatch):
    built = extension._load_extension()
    binary = Path(built.__file__)
    binary.write_bytes(b"corrupt binary")
    monkeypatch.setattr(extension, "_module", None)
    original = cpp_extension.load

    def rebuild(**kwargs):
        assert not binary.exists(), "Ninja must relink, rather than accept a corrupt cached output"
        return original(**kwargs)

    monkeypatch.setattr(cpp_extension, "load", rebuild)
    extension._load_extension()
    assert len(fake_build) == 2


@pytest.mark.parametrize("receipt", ["missing", "invalid", "different", "escaping"])
def test_incomplete_cache_is_not_loaded(tmp_path, receipt):
    identity = {"source": "current"}
    if receipt == "invalid":
        (tmp_path / "build.json").write_text("{")
    elif receipt != "missing":
        (tmp_path / "build.json").write_text(json.dumps({
            "identity": identity if receipt == "escaping" else {"source": "old"},
            "binary": "../outside.so", "binary_sha256": "bad",
        }))
    assert extension._cached_module(tmp_path, "native", identity) is None


def test_cpu_only_torch_is_rejected_before_compilation():
    cpu = SimpleNamespace(version=SimpleNamespace(cuda=None))
    with pytest.raises(RuntimeError, match="CPU-only"):
        extension._build_identity(cpu, None)


def test_missing_toolkit_has_actionable_error():
    fake_torch = SimpleNamespace(version=SimpleNamespace(cuda="12.6"))
    with pytest.raises(RuntimeError, match="also requires NVCC"):
        extension._build_identity(fake_torch, SimpleNamespace(CUDA_HOME=None))


def test_cuda_major_mismatch_is_rejected(monkeypatch):
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.9")
    monkeypatch.setattr(extension, "_compiler_identity", lambda *_: {"version": "release 13.0"})
    fake_torch = SimpleNamespace(version=SimpleNamespace(cuda="12.6"))
    with pytest.raises(RuntimeError, match="same CUDA major version"):
        extension._build_identity(fake_torch, SimpleNamespace(CUDA_HOME="/cuda"))


def test_missing_compiler_has_actionable_error(monkeypatch):
    monkeypatch.setattr(extension.shutil, "which", lambda *_: None)
    with pytest.raises(RuntimeError, match="developer terminal"):
        extension._compiler_identity("cl", [])


def test_capture_requires_warmup(fake_build, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        extension._load_extension()
    assert not fake_build


def test_header_changes_invalidate_source_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(extension, "_SOURCE_DIR", tmp_path)
    (tmp_path / "kernel.cu").write_text("kernel")
    (tmp_path / "geometry.cuh").write_text("before")
    before = extension._source_hash()
    (tmp_path / "geometry.cuh").write_text("after")
    assert extension._source_hash() != before
