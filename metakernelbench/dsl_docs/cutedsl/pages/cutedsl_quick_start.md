<!-- source: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html -->

<a id="quick-start-guide"></a>
# Quick Start Guide

<a id="compatibility-requirements"></a>
## Compatibility Requirements

The CUTLASS DSL 4.4 release currently supports **Linux** and **Python 3.10 - 3.14** only.

Only Linux x86\_64 and aarch64 are supported. Additional platform support will be added in future releases.

CUTLASS DSL supports the same NVIDIA driver version as the corresponding CUDA Toolkit
(CUDA Toolkit 12.9 or CUDA Toolkit 13.1). Specifically, for 12.9, the driver version must be 575.51.03 or later.

<a id="installation"></a>
## Installation

To ensure compatibility with the examples and code on GitHub,
use the setup.sh file from the corresponding commit in the repository.

```bash
git clone https://github.com/NVIDIA/cutlass.git

# For CUDA Toolkit 12.9:
./cutlass/python/CuTeDSL/setup.sh --cu12

# For CUDA Toolkit 13.1:
./cutlass/python/CuTeDSL/setup.sh --cu13
```

If you just want to try out the last known stable release of the CUTLASS DSL (may not be compatible with the latest examples and code), run:

```bash
# For CUDA Toolkit 12.9:
pip install nvidia-cutlass-dsl

# For CUDA Toolkit 13.1:
pip install "nvidia-cutlass-dsl[cu13]"
```

The `nvidia-cutlass-dsl` wheel includes everything needed to generate GPU kernels.

<a id="recommended-dependencies"></a>
## Recommended Dependencies

To run examples and begin development, we recommend installing:

```bash
pip install torch jupyter mypy==1.19.1
```

<a id="recommended-python-environment-variables-for-jupyter-notebooks"></a>
## Recommended Python environment variables for jupyter notebooks

We recommend setting the following environment variable when running jupyter notebooks.

```bash
export PYTHONUNBUFFERED=1
```
