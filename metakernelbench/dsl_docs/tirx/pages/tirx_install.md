<!-- source: https://tvm.apache.org/docs/tirx/install.html -->

<a id="installation"></a>
# Installation

There are two pieces:

- the **TIRx compiler** (`tvm.tirx`), which ships inside Apache TVM — this is
  all you need to write and compile kernels;
- the optional **kernel library** (`tirx-kernels`), a set of ready-made GEMM
  and attention kernels built with TIRx.

<a id="requirements"></a>
## Requirements

- Python ≥ 3.10.
- An NVIDIA GPU with a recent CUDA toolkit. The bundled kernels target Blackwell
  (`sm_100a`); the compiler itself targets GPUs and accelerators more broadly.

<a id="install-the-tirx-compiler"></a>
## Install the TIRx compiler

Install the Apache TVM wheel (the TIRx compiler is the `tvm.tirx` module):

```bash
pip install apache-tvm
```

Verify:

```bash
python -c "import tvm, tvm.tirx; print(tvm.__version__)"
```

<a id="install-the-kernel-library-optional"></a>
## Install the kernel library (optional)

`tirx-kernels` provides prebuilt kernels (`fp16_bf16_gemm`,
`fp8_blockwise_gemm`, `nvfp4_gemm`, `flash_attention4`). It has no PyPI
wheel — install it from source:

```bash
git clone https://github.com/mlc-ai/tirx-kernels
cd tirx-kernels
pip install -e .
```

Its runtime dependencies are **not** pulled from PyPI and must be available
separately (they are imported lazily, so `import tirx_kernels` and kernel
discovery work without them — they are only needed to actually compile/run a
kernel):

| Dependency | Needed by | Notes |
| --- | --- | --- |
| `tvm.tirx` | all kernels | the TIRx compiler (installed above, or put a source checkout’s `python/` on `PYTHONPATH`) |
| `torch` | all kernels | a CUDA build matching your GPU |
| `deep_gemm` | `fp8_blockwise_gemm` | optional — quantization helpers and the reference baseline |
| `flashinfer` | `nvfp4_gemm` | optional — quantization and the baseline |
