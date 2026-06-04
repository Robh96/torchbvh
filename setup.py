import os
import sys

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _detect_gencode_flag():
    import torch

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
    else:
        arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
        if arch_list:
            first = arch_list.split()[0].split(";")[0]
            parts = first.split(".")
            major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        else:
            print(
                "WARNING: No CUDA device found and TORCH_CUDA_ARCH_LIST is not set. "
                "Defaulting to sm_80."
            )
            major, minor = 8, 0

    arch = f"{major}{minor}"
    return f"-gencode=arch=compute_{arch},code=sm_{arch}"


setup(
    name="torchbvh",
    version="0.0.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="torchbvh._C",
            sources=[
                "torchbvh/csrc/bindings.cpp",
                "torchbvh/csrc/bvh_build.cu",
                "torchbvh/csrc/fps_sample.cu",
                "torchbvh/csrc/knn_query.cu",
                "torchbvh/csrc/mls_fused.cu",
                "torchbvh/csrc/morton_sort.cu",
                "torchbvh/csrc/smoke.cu",
            ],
            include_dirs=["torchbvh/csrc"],
            extra_compile_args={
                "cxx": ["/O2"] if sys.platform == "win32" else ["-O2"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    _detect_gencode_flag(),
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
