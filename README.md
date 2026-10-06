# MetaKernelBench: Measuring GPU Kernel Knowledge Transfer Beyond Code

[![arXiv](https://img.shields.io/badge/arXiv-2610.05014-b31b1b.svg)](https://arxiv.org/abs/2610.05014)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://yige24.github.io/MetaKernelBench)

[Xueyi Chen](https://yige24.github.io)<sup>1,\*</sup>, Shiyu Liu<sup>1,2,\*</sup>, [Xin Jin](https://jinxins.github.io/)<sup>1</sup>, [Yuhua Zheng](https://pillor9.github.io/)<sup>1,3</sup>, [Xin Li](https://profile.slowist.top/)<sup>1,3</sup>, [Haolei Bai](https://deadlykitten4.github.io/)<sup>1</sup>, [Junhan Zhu](https://alrightlone.github.io/)<sup>1</sup>, [Huan Wang](https://huanwang.tech/)<sup>1,†</sup>

<sup>1</sup>Westlake University &nbsp; <sup>2</sup>Shanghai Jiao Tong University &nbsp; <sup>3</sup>Zhejiang University

<sup>\*</sup>Equal contribution &nbsp; <sup>†</sup>Corresponding author

---

## Abstract

> Recent GPU kernel optimization agents retain what they learn in knowledge bases or as distilled skills. Kernel benchmarks score each attempt's implementation for correctness and speed but leave the reuse value of retained experience unmeasured. We introduce MetaKernelBench, which measures whether experience distilled from an attempt in one kernel domain-specific language (DSL) improves a fresh attempt at the same problem in another. Its 74 problems are fused subgraphs in six families, each posed as a pair of CuTe DSL and TIRx variants that differ only in the DSL. The agent first attempts each variant solo and is instructed to distill what it learns into a natural-language skill, which is transferred whether or not the source attempt passes verification. The skill is the only extra input to a skill-conditioned attempt by the same model in the other DSL. We compare each skill-conditioned attempt with the solo attempt on the same variant under matched per-attempt budgets, scoring correctness and end-to-end runtime. Across six models and both directions, paired lift over solo attempts ranges from -19% to +29%. Four models gain in both directions, yet regressions occur on 16% to 45% of problems in every model and direction. Outcomes follow the source attempt's result relative to the target's solo attempt rather than source success alone, improving in 71% of comparisons when the source stands above and regressing in 54% when it stands below. MetaKernelBench complements implementation-quality metrics by measuring same-problem cross-DSL kernel knowledge transfer.

## Requirements

- Python 3.12
- A [Modal](https://modal.com) account with access to B200 GPUs; sandboxes and grading run on Modal
- API keys for the model providers you run

## Setup

```bash
cd MetaKernelBench
uv venv --python 3.12
source .venv/bin/activate
uv pip install "harbor[modal]==0.20.0" python-dotenv pyarrow
modal setup
cat > .env <<'EOF'
OPENROUTER_API_KEY=
DEEPSEEK_API_KEY=
DASHSCOPE_API_KEY=
EOF
```

`metakernelbench.cli` loads `.env` from the repository root; fill in the keys for the providers you use.

## Problem set

The repository ships 32 of the 74 problems. The other 42 derive from the [nvidia/SOL-ExecBench](https://huggingface.co/datasets/nvidia/SOL-ExecBench) dataset, whose license forbids redistribution, so they are assembled on your machine from your own copy of the dataset:

```bash
python -m metakernelbench.sol_execbench                       # downloads the dataset from Hugging Face
python -m metakernelbench.sol_execbench --source modelscope   # or from the ModelScope mirror
python -m metakernelbench.sol_execbench --source /path/to/dir # or from L1.parquet, L2.parquet, Quant.parquet you already have
```

The dataset is subject to the NVIDIA Evaluation Dataset License Agreement (`licenses/sol-execbench-dataset.LICENSE`). Each assembled file is checked against its recorded hash. The assembled files are covered by that agreement, not by this repository's license, and are ignored by git.

## Run

```bash
python -m metakernelbench.build                                                               # run first
python -m metakernelbench.cli --model openrouter/z-ai/glm-5.3 --trials 'p32_r1_solo_cutedsl'  # one trial
python -m metakernelbench.cli --model openrouter/z-ai/glm-5.3                                 # one model, everything
./run.sh all glm all --n-replicates 1                                                         # same, with aliases; see ./run.sh --help
```

A trial is addressed as `p<NN>_r<k>_<arm>`: problem number, replicate, and one of `solo_cutedsl`, `solo_tirx`, `cutedsl2tirx_skill`, `tirx2cutedsl_skill`.

Results accumulate under `results/<model>/`; a repeated invocation reuses finished trials and reruns failed ones, so several partial runs equal one full run. `--dry-run` prints the plan without running anything. `results/<model>/results.json` is the current snapshot and carries a legend describing its fields. See `--help` for the other options.

## Citation

```bibtex
@article{chen2026metakernelbench,
  title   = {MetaKernelBench: Measuring GPU Kernel Knowledge Transfer Beyond Code},
  author  = {Chen, Xueyi and Liu, Shiyu and Jin, Xin and Zheng, Yuhua and Li, Xin and Bai, Haolei and Zhu, Junhan and Wang, Huan},
  journal = {arXiv preprint arXiv:2610.05014},
  year    = {2026}
}
```

## License

The code and the problems in this repository are licensed under the Apache License 2.0 (`LICENSE`), except for the third-party materials listed in `licenses/README.md`, which keep their own licenses.
