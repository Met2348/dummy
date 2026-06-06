# ERIC-3080Ti Local Migration Plan

> Branch: `ERIC-3080Ti/env-migration`
>
> Logical machine label: `ERIC-3080Ti`
>
> Windows host name observed from `COMPUTERNAME`: `ERIC-12900H-308`
>
> Date: 2026-06-06

## Goal

Make this repository runnable on the local ERIC-3080Ti machine with a reproducible environment, a clear test matrix, and documented Windows-native limitations. The working rule is:

1. Core educational code and tests should run locally.
2. Linux/WSL2-only stacks should be detected and documented instead of silently failing.
3. Optional heavyweight dependencies should not be required for smoke tests unless the module truly needs them.
4. All environment fixes should live on this branch and be pushed to `origin`.

## Current Local Baseline

| Item | Observed |
|---|---|
| Git base | `main...origin/main`, clean before branch creation |
| Working branch | `ERIC-3080Ti/env-migration` |
| Remote | `origin https://github.com/Met2348/dummy.git` |
| Default `python` | `D:\Anaconda\python.exe`, Python 3.9.13 |
| Default `py` | Python 3.14.3 |
| Available Python 3.13 | Python 3.13.9 from Windows Python launcher |
| Existing conda 3.10 | `D:\Anaconda\envs\common310`, Python 3.10.6 |
| GPU | NVIDIA GeForce RTX 3080-class GPU, 16 GB VRAM |
| Driver/CUDA runtime | Driver 595.97, CUDA 13.2 reported by `nvidia-smi` |

The default Anaconda environment is not suitable as the repo baseline: it currently has old ML packages such as `torch==1.12.1`, `transformers==4.22.1`, `numpy==1.21.5`, and no `peft`, `accelerate`, `triton`, `vllm`, `flash_attn`, or `bitsandbytes`.

## Repository Shape

| Category | Count |
|---|---:|
| Learning modules | 46 |
| `environment/requirements.txt` files | 39 |
| `environment/verify_env.py` files | 39 |
| Test files | 110 |

Seven Module 8 infra topics do not have `environment/verify_env.py`: `gpu-architecture`, `cuda-essentials`, `kernel-engineering`, `cluster-networking`, `storage-dataops`, `training-orchestration`, and `infra-graduation`.

## Dependency Risk Tiers

### Tier A: Local-first / stdlib

These modules are expected to run on Windows native with no external packages beyond Python:

- `agent-code-eval`
- `agent-foundations`
- `agent-framework-stack`
- `agent-graduation`
- `agent-memory-context`
- `eval-foundations`
- `eval-graduation`
- `llm-judge-arena`
- `multi-agent-orchestration`
- `rag-essential`
- `reasoning-eval`
- `red-team-jailbreak`
- `safety-defense`
- `tool-use-mcp`

### Tier B: CPU or CUDA PyTorch, Windows-native expected

These should run locally after installing a modern PyTorch stack and common scientific packages:

- `prompt-tuning-family`
- `lora-family`
- `adapter-tuning-family`
- `data-curation`
- `transformer-deep`
- `moe-architecture`
- `ssm-hybrid`
- `long-context`
- `pretraining-recipe`
- `small-model-graduation`
- `dpo-family`
- `process-reward`
- `rl-foundations`
- `rlhf-classic`
- `scaling-infra`
- `speculative-decoding`
- `production-serving`
- `serving-graduation`

### Tier C: GPU/Linux/WSL2-heavy or optional production stacks

These include dependencies that may not install or run correctly on Windows native and should be handled with skip/detect logic:

- `inference-engine-core`: `vllm`, `flash-attn`, `triton`
- `reasoning-r1`: `verl`, `vllm`, Ray/GRPO stack
- `rl-sota-2026`: `verl`, `vllm`, reward-model utilities
- `distributed-inference`: Ray serving / multi-GPU assumptions
- `quantization-deploy`: `auto-gptq`, `autoawq`, `bitsandbytes`
- `sglang-radixattention`: `sglang`, `outlines`, `xgrammar`
- `multimodal-agent`: browser/VLM/benchmark extras

For Tier C, the ERIC-3080Ti branch should prefer graceful detection and minimal/mock tests over unconditional import failures.

## Execution Plan

1. Create an isolated repo environment at `.venv` and never rely on base Anaconda.
2. Install a minimal modern test stack first: `pip`, `setuptools`, `wheel`, `pytest`, `numpy`, `matplotlib`.
3. Install PyTorch with CUDA support and verify `torch.cuda.is_available()` on the 3080-class GPU.
4. Install the common LLM stack in controlled batches: `transformers`, `tokenizers`, `accelerate`, `datasets`, `peft`, `einops`, `sentencepiece`, `tiktoken`.
5. Run all stdlib-only module tests first to establish a clean baseline.
6. Run PyTorch module tests next and fix portability issues.
7. Run environment verifiers and downgrade hard failures to skips where the dependency is documented as optional or WSL2-only.
8. Produce a test matrix with pass/fail/skip reasons.
9. Commit environment docs, runner scripts, and portability fixes.
10. Push `ERIC-3080Ti/env-migration` to `origin`.

## Acceptance Criteria

- A local environment can be recreated from documented commands.
- All Tier A tests pass.
- Tier B tests pass or have specific, documented reasons with fixes applied where appropriate.
- Tier C modules do not fail opaquely on Windows native; unsupported stacks are reported clearly.
- `git status` is clean after commit.
- Branch `ERIC-3080Ti/env-migration` is pushed to `origin`.
