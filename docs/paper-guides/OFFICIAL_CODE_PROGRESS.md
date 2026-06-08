# ERIC-3080Ti Official Code Progress

> Branch: `ERIC-3080Ti/paper-guides`
>
> Machine target: `ERIC-3080Ti`
>
> Started: 2026-06-09

## Current Count

- Processed papers: 6 / 46
- Official submodules/artifacts added: 6
- Teaching notebooks added: 6
- Smoke checks passed: 6
- Notebook execution checks passed: 6 / 6
- Existing local test suites passed in this batch: 5
- Heavy existing test deferred: 1

## Batch 01: Papers 01-06

| # | Module | Official source | Pinned commit | Notebook | Verification |
|---:|---|---|---|---|---|
| 01 | `transformer-deep` | `tensorflow/tensor2tensor` | `bafdc1b` | `notebooks/01_attention_walkthrough.ipynb` | attention mechanism smoke passed |
| 02 | `dpo-family` | `eric-mitchell/direct-preference-optimization` | `f8b8c0f` | `notebooks/01_dpo_walkthrough.ipynb` | DPO smoke passed; `test_dpo_loss_equivalence.py`: 4 passed |
| 03 | `lora-family` | `microsoft/LoRA` | `c4593f0` | `notebooks/01_lora_walkthrough.ipynb` | LoRA mechanism smoke passed; `test_lora_consistency.py` deferred after 120s timeout because it builds/downloads GPT-2 + PEFT path |
| 04 | `rlhf-classic` | `openai/following-instructions-human-feedback` | `5c0534c` | `notebooks/01_instructgpt_walkthrough.ipynb` | RLHF smoke passed; `test_three_stage_pipeline.py`: 7 passed |
| 05 | `rl-foundations` | `openai/baselines` | `ea25b9e` | `notebooks/01_ppo_walkthrough.ipynb` | PPO smoke passed; `test_ppo_consistency.py`: 6 passed |
| 06 | `reasoning-r1` | `deepseek-ai/DeepSeek-R1` | `0cf7856` | `notebooks/01_deepseek_r1_walkthrough.ipynb` | DeepSeek-R1/GRPO smoke passed; `reasoning-r1/src/tests`: 31 passed |

## Notes

- Official code is added as git submodules under `learning/<module>/official/repos/<repo>`.
- Each module has `official/README.md` explaining official status, reproduction level, key files to read, local setup, and notebook path.
- Teaching notebooks are runnable from the repository root and are designed to avoid large model downloads by default.
- All six batch-01 notebooks were executed top-to-bottom with a lightweight local executor after clearing outputs.
- Old TensorFlow-era repositories are treated as source-inspection artifacts unless a clean local environment is explicitly prepared.

## Next Paper

Continue with:

- #07 `kernel-engineering`: FlashAttention official repo `Dao-AILab/flash-attention`, likely WSL2/Linux CUDA for full package tests, CPU-safe notebook should teach online softmax and tiling first.
