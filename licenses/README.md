# Third-party licenses

MetaKernelBench is licensed under the Apache License 2.0 (`../LICENSE`). The materials below keep their own licenses, reproduced in this directory.

| Material | Upstream | License | Files |
|---|---|---|---|
| Problems 01–09, 14–18, 28, 36, 44, 71–73 | [FlashInfer-Trace](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace) and [FlashInfer-Bench](https://github.com/flashinfer-ai/flashinfer-bench) | Apache-2.0 | `flashinfer-bench.LICENSE`, `flashinfer-bench.NOTICE` |
| Problems 19–26 | [vLLM](https://github.com/vllm-project/vllm) at commit b9b6306e | Apache-2.0 | `vllm.LICENSE` |
| Problems 68–70, 74 | [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | MIT | `flash-linear-attention.LICENSE` |
| `metakernelbench/dsl_docs/cutedsl/` | [CUTLASS](https://github.com/NVIDIA/cutlass) `media/docs/pythonDSL` | BSD-3-Clause | `cutlass.LICENSE` |
| `metakernelbench/dsl_docs/tirx/` | [Apache TVM](https://github.com/apache/tvm) `docs/tirx` | Apache-2.0 | `tvm.LICENSE`, `tvm.NOTICE` |
| Problems 10–13, 27, 29–35, 37–43, 45–67 (assembled locally, not distributed) | [nvidia/SOL-ExecBench](https://huggingface.co/datasets/nvidia/SOL-ExecBench) dataset | NVIDIA Evaluation Dataset License Agreement | `sol-execbench-dataset.LICENSE` |

The SOL-ExecBench dataset may not be redistributed, so this repository ships no part of it. `problems/_sol_execbench/` holds none of the dataset's text: every span taken from the dataset is cut out of each problem file and recorded as a position in the dataset's reference code, and entry names and workload identifiers are recorded as indices. `python -m metakernelbench.sol_execbench` fills those spans back in from your own copy of the dataset, which you download under its license. The assembled files are covered by the NVIDIA agreement, not by this repository's Apache-2.0 license, and must not be committed or redistributed; the Apache-2.0 license likewise does not extend to anything in the recipes that derives from the dataset.

The CuTe DSL software (`nvidia-cutlass-dsl`) is not part of this repository; the sandbox installs it from PyPI under NVIDIA's own license.
