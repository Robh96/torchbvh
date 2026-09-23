import sys
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


setup(
    name="torchbvh",
    version="0.3.1",
    description="GPU-native BVH, k-NN, ray tracing, MLS interpolation, and FPS primitives for PyTorch.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Robh96/torchbvh",
    license="MIT",
    project_urls={
        "Documentation": "https://torchbvh.readthedocs.io/",
        "Source": "https://github.com/Robh96/torchbvh",
    },
    packages=find_packages(include=["torchbvh", "torchbvh.*"]),
    python_requires=">=3.10",
    install_requires=["torch>=2.0"],
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
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
                "torchbvh/csrc/ray_query.cu",
            ],
            include_dirs=["torchbvh/csrc"],
            extra_compile_args={
                "cxx": ["/O2"] if sys.platform == "win32" else ["-O2"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
