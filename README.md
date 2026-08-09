# Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models


<p align="center">
<a href="https://arxiv.org/pdf/2601.18734v3"><img src="https://img.shields.io/badge/arXiv-2601.18734-b31b1b.svg"></a>
<a href="https://siyan-zhao.github.io/blog/2026/opsd/"><img src="https://img.shields.io/badge/Blog-Post-blue.svg"></a>
</p>

---
## Overview

**On-Policy Self-Distillation (OPSD)** trains a single model to act as both student and teacher by conditioning on different contexts — the student sees only the problem, while the teacher additionally sees the ground-truth solution — and performs token-level distribution matching along the student's own on-policy trajectories.

This repository also supports **On-Policy Distillation (OPD)** through the same
training entry point. OPD uses Qwen3-8B as a fixed external teacher and scores
the student's on-policy completion with the same prompt and condition,
\(\pi_T(y\mid x)\), without exposing the ground-truth solution to the teacher.


## Updates

- **Mar 18, 2026**: Released updated code. 

  (1) Fixed chat template and zero2 bugs (see [template issue](https://github.com/huggingface/trl/issues/5241)), we re-ran experiments with updated results (detailed results & ablations updated on arxiv/blog). The fixes yield improved OPSD performance, most notably on Qwen3-1.7B.

  (2) Added a new training stabilization strategy 🚀: per-token point-wise KL clipping. We find style tokens (such as 'wait', 'think') can exhibit 6–15× higher KL divergence than math-related tokens, and dominates the training signal. Clipping stablizes training and improves performance.


-  **Mar 3, 2026**: Initial code release.

## Installation


```bash
bash scripts/setup_env.sh
conda activate "$PWD/.cache/conda/opsd"
```
If you encounter difficulties installing flash-attn, you can check the version matching your CUDA and PyTorch versions from the [flash-attention releases page](https://github.com/Dao-AILab/flash-attention/releases).

The code uses `trl`'s experimental GOLD trainer as a base.

## Portable 1-node reproduction

The scripts below are designed for a fresh GitHub checkout. Prepared datasets,
checkpoints, logs, and caches stay below the repository root. Base-model weights
remain in the Hugging Face cache and are exposed through links in `models/`.

When the target platform already provides the environment, prepare the pinned
training data, five evaluation datasets, and Hugging Face model links:

```bash
bash scripts/prepare_train_data.sh
bash scripts/prepare_eval_data.sh
bash scripts/prepare_models.sh
```

The combined data command is equivalent and prepares both scopes:

```bash
bash scripts/prepare_data.sh
```

All preparation commands are idempotent: existing files are validated against
their pinned revisions, row counts, and required fields before downloading.

Model preparation defaults to all three models. A single model can also be
selected with the positional model argument:

```bash
bash scripts/prepare_models.sh 1.7b
bash scripts/prepare_models.sh 4b
bash scripts/prepare_models.sh 8b
```

Run the paper's main OPSD configuration:

```bash
bash scripts/OPSD/run_opsd_1b.sh   # 8 GPUs, 100 steps, effective batch size 32
bash scripts/OPSD/run_opsd_4b.sh   # 8 GPUs, 100 steps, effective batch size 32
bash scripts/OPSD/run_opsd_8b.sh   # 8 GPUs, 100 steps, effective batch size 32
```

These scripts explicitly use the paper's main training modes: the student
rollout is thinking-mode off and the privileged teacher is thinking-mode on.
The model, optimizer, generation, and effective-batch settings are unchanged;
the launch topology uses all 8 GPUs and preserves effective batch size 32.
WandB defaults to offline mode for unattended runs; use `WANDB_MODE=online`
when desired.

Run OPD with either supported student size:

```bash
bash scripts/OPD/run_opd_1b.sh   # Qwen3-1.7B student, Qwen3-8B teacher
bash scripts/OPD/run_opd_4b.sh   # Qwen3-4B student, Qwen3-8B teacher
```

The optimization, generation, and colocated-vLLM settings mirror the
corresponding OPSD launchers and target an 8×H100-80GB node. There is
intentionally no Qwen3-8B OPD student launcher.

After training exits successfully, each main `run_opsd_*.sh` launcher
automatically starts its matching five-dataset thinking-mode evaluation. Set
`AUTO_EVAL=0` to train without the post-training evaluation. Evaluation scope
can be adjusted with `CHECKPOINTS`, `DATASETS`, `VAL_N`, and the variables
shown below.

Evaluate checkpoints 25/50/75/100 on AIME24, AIME25, BeyondAIME, HMMT25,
and AMO-Bench:

```bash
bash scripts/run_eval.sh 1.7b
bash scripts/run_eval.sh 4b
bash scripts/run_eval.sh 8b
```

Formal evaluation uses thinking mode, 12 samples per problem, temperature 1.0,
top-p 0.95, disabled top-k, and 38,912 maximum new tokens. Detailed generations
and aggregate `summary.json`/`summary.csv` files are written under
`outputs/eval/<model>/`. `Pass@12` is the fraction of problems with at least one
correct sample; `Avg@12` is accuracy over all 12 samples per problem.
After all checkpoints finish, `best_by_dataset.json` and
`best_by_dataset.csv` select the highest-Avg@12 checkpoint independently for
each dataset and report the macro average, matching the paper's Table 2
reporting convention.

Long runs are resumable: completed result files are skipped. The evaluation
scope can be narrowed without editing a script, for example:

```bash
CHECKPOINTS="25 100" DATASETS="aime24 beyond-aime" \
    bash scripts/run_eval.sh 8b
```

Set `HF_HOME` before model preparation if the target platform keeps its model
cache on a dedicated volume. The resulting `models/Qwen3-*` links are checked
before any download is attempted.

The pinned PyTorch build uses CUDA 12.8. The NVIDIA driver may be newer, but
the `nvcc` toolkit used to build DeepSpeed/FlashAttention extensions should be
CUDA 12.8 as well; `setup_env.sh` prints a warning when it detects a mismatch.

## Repository Structure

```
├── opsd_trainer.py          # OPSDTrainer: core self-distillation trainer
├── data_collator.py         # Data collator for self-distillation
├── opsd_train.py            # OPSD training entry point
├── sft_train.py             # SFT baseline training entry point
├── grpo_train.py            # GRPO baseline training entry point
├── accelerate.yaml          # Accelerate config (multi-GPU)
├── scripts/
│   ├── prepare_all.sh       # Pinned local data + HF model links
│   ├── OPSD/               # OPSD launchers (1.7B, 4B, and 8B students)
│   ├── OPD/                # OPD launchers (1.7B and 4B students; 8B teacher)
│   ├── prepare_models.sh    # Prepare all models or one selected model
│   ├── run_eval.sh          # Model-parameterized formal evaluation
│   ├── run_sft.sh           # SFT baseline launcher
│   └── run_grpo.sh          # GRPO baseline launcher
└── eval/
    ├── evaluate_math.py     # Evaluation script (vLLM)
    ├── run_model_eval.sh    # Model/checkpoint/dataset matrix runner
    ├── summarize_results.py # JSON/CSV aggregate metrics
    └── run_eval.sh          # Example evaluation script
```

## Quick Start

Reproduce results on Qwen3-1.7B with the portable single-node 8-GPU launcher:

```bash
bash scripts/OPSD/run_opsd_1b.sh
```
Evaluation: (evaluation takes ~ 30-50 minutes on 4xh100 for each checkpoint) 
```bash
cd eval
bash run_eval.sh
```

### Evaluation Results across Tasks on Qwen3-1.7B

<div align="center">
<table>
<tr>
<th align="center">AIME24</th>
<th align="center">AIME25</th>
<th align="center">HMMT25</th>
</tr>
<tr>
<td>

| Step | Avg@12 |
|---|---|
| Base | 51.5% |
| 25 | 51.4% |
| 50 | 52.8% |
| 75 | 54.4% |
| 100 | 57.2% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 36.7% |
| 25 | 42.5% |
| 50 | 43.9% |
| 75 | 40.6% |
| 100 | 41.1% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 23.1% |
| 25 | 24.7% |
| 50 | 27.8% |
| 75 | 26.9% |
| 100 | 29.2% |

</td>
</tr>
</table>
</div>

> **Evaluation settings:** temperature=1.0, thinking mode enabled, max new tokens=38912, top-p=none, top-k disabled, min-p=0, presence penalty=0, num samples=12


## Non-Thinking Mode

OPSD can also run in non-thinking setting where both the Qwen student and teacher are enabled_thinking=False during training (`--student_thinking False --teacher_thinking False`) and evaluated with non-thinking inference (`--no_thinking`), with faster evaluation time than thinking mode.

Training:
```bash
bash scripts/run_opsd_4b_nonthink.sh
bash scripts/run_opsd_8b_nonthink.sh
```

Evaluation:
```bash
cd eval
bash run_eval_nonthink.sh
```

### Evaluation Results with Non-Thinking Mode across Models

#### Qwen3-8B (`--jsd_token_clip 1e-7`)

<div align="center">
<table>
<tr>
<th align="center">AIME24</th>
<th align="center">AIME25</th>
<th align="center">HMMT25</th>
</tr>
<tr>
<td>

| Step | Avg@12 |
|---|---|
| Base | 26.4% |
| 50 | 49.7% |
| 75 | 45.3% |
| 100 | 38.3% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 19.7% |
| 50 | 35.0% |
| 75 | 26.9% |
| 100 | 27.5% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 10.8% |
| 50 | 18.3% |
| 75 | 17.5% |
| 100 | 15.3% |

</td>
</tr>
</table>
</div>

#### Qwen3-4B (`--jsd_token_clip 1e-6`)

<div align="center">
<table>
<tr>
<th align="center">AIME24</th>
<th align="center">AIME25</th>
<th align="center">HMMT25</th>
</tr>
<tr>
<td>

| Step | Avg@12 |
|---|---|
| Base | 23.1% |
| 50 | 20.3% |
| 75 | 27.5% |
| 100 | 31.1% |
| 150 | 32.8% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 21.4% |
| 50 | 21.4% |
| 75 | 20.8% |
| 100 | 21.1% |
| 150 | 21.9% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 10.8% |
| 50 | 11.1% |
| 75 | 13.1% |
| 100 | 16.4% |
| 150 | 14.4% |

</td>
</tr>
</table>
</div>

#### Qwen3-1.7B (`--jsd_token_clip 1e-6`)

<div align="center">
<table>
<tr>
<th align="center">AIME24</th>
<th align="center">AIME25</th>
<th align="center">HMMT25</th>
</tr>
<tr>
<td>

| Step | Avg@12 |
|---|---|
| Base | 11.9% |
| 50 | 15.0% |
| 75 | 13.9% |
| 100 | 12.5% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 9.2% |
| 50 | 6.2% |
| 75 | 8.3% |
| 100 | 8.1% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 5.0% |
| 25 | 7.2% |
| 50 | 5.8% |
| 75 | 5.0% |

</td>
</tr>
</table>
</div>

> **Evaluation settings:** temperature=1.0, non-thinking mode, num samples=12.



## Key distillation arguments

| Argument | Default | Description |
|---|---|---|
| `--alg` | `opsd` | Select `opsd` for the privileged self-teacher or `opd` for the fixed external Qwen3-8B teacher. OPD accepts only Qwen3-1.7B and Qwen3-4B students. |
| `--teacher_model_name_or_path` | `models/Qwen3-8B` for OPD | External teacher used by OPD. Its architecture is validated as Qwen3-8B. This argument is rejected for OPSD. |
| `--fixed_teacher` | `False` | Fix the teacher to the initial policy (step 0). Requires --use_peft. Note ❗ If you disable PEFT, the teacher will keep updating at every training step, which may make training unstable. Our main results use the fixed teacher, which is currently implemented with LoRA adapter weights. |
| `--use_tinker_loss` | `False` | Use sampled-token policy-gradient objective instead of full-vocabulary JSD. More memory efficient. Currently no clipped implemented for this variant, could be unstable. |
| `--max_completion_length` | — | Student generation length for distillation. We use 1024 in our main experiments. |
| `--beta` | — | Interpolation weight for the JSD mixture distribution. Beta=0 means forward KL and 1 means reverse KL. |
| `--jsd_token_clip` | 0.05 | Clip the JSD loss for each token to a maximum value. This can improve stability by preventing stylistic tokens from dominating the training signal. Note when clipping is applied, the loss can be negative due to positive KL summand being capped. | 
| `--reason_first` | `False` | Prepend an explicit rationalization to the teacher context before distillation. |
| `--run_config` | `None` | Custom name suffix for the output directory and WandB run. |

### SFT Baseline

See [`scripts/run_sft.sh`](scripts/run_sft.sh).

### GRPO Baseline

See [`scripts/run_grpo.sh`](scripts/run_grpo.sh).

### Acknowledgements
Our implementation builds on [TRL GOLD Trainer](https://huggingface.co/docs/trl/gold_trainer). We sincerely thank [@simran135](https://github.com/simran135) and [@beanie00](https://github.com/beanie00) for identifying the prompt template bugs and the zero-2 issue, respectively!

## Citation
If you find this useful, please consider citing:
```bibtex
@article{zhao2026self,
  title={Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models},
  author={Zhao, Siyan and Xie, Zhihui and Liu, Mengchen and Huang, Jing and Pang, Guan and Chen, Feiyu and Grover, Aditya},
  journal={arXiv preprint arXiv:2601.18734},
  year={2026}
}
```
