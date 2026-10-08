from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


setup(
    name="torchbvh",
    version="0.3.3",
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
    # PyTorch is supplied by the user, so pip never selects a replacement build.
    install_requires=["filelock>=3.12", "ninja>=1.11"],
    package_data={"torchbvh": ["csrc/*.cpp", "csrc/*.cu", "csrc/*.cuh"]},
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
)
