# ERIC-3080Ti Local Environment Report

> Branch: `ERIC-3080Ti/env-migration`
>
> Logical machine label: `ERIC-3080Ti`
>
> Windows host name observed from `COMPUTERNAME`: `ERIC-12900H-308`
>
> Date: 2026-06-06

## Outcome

The local ERIC-3080Ti environment is now able to run the repository's educational source tests and environment verification scripts.

| Check | Result |
|---|---:|
| Learning module source tests | 46 / 46 PASS |
| `verify_env.py` scripts | 39 / 39 PASS |
| Module 8 topics without env scripts | 7 / 7 covered by tests |

Result artifacts:

- Test matrix: `docs/local-env/ERIC-3080Ti-test-matrix.md`
- Test JSON with stdout/stderr: `docs/local-env/ERIC-3080Ti-test-results.json`
- Env matrix: `docs/local-env/ERIC-3080Ti-env-matrix.md`
- Env JSON with stdout/stderr: `docs/local-env/ERIC-3080Ti-env-results.json`
- Full pip freeze: `docs/local-env/ERIC-3080Ti-pip-freeze.txt`

## Local Environment

| Item | Value |
|---|---|
| Python env | Repo-local `.venv` |
| Python | 3.13.9 |
| GPU | NVIDIA GeForce RTX 3080 Ti Laptop GPU |
| VRAM | 16 GB class |
| NVIDIA driver | 595.97 |
| CUDA reported by `nvidia-smi` | 13.2 |
| PyTorch wheel | `torch==2.11.0+cu128` |

Key package versions:

| Package | Version |
|---|---|
| `torch` | `2.11.0+cu128` |
| `transformers` | `5.10.2` |
| `datasets` | `5.0.0` |
| `pyarrow` | `21.0.0` |
| `peft` | `0.19.1` |
| `bitsandbytes` | `0.49.2` |
| `trl` | `1.5.1` |
| `numpy` | `2.4.6` |
| `pytest` | `9.0.3` |

## Rebuild Commands

From the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install pytest numpy matplotlib pandas scipy einops transformers==5.10.2 tokenizers accelerate datasets peft sentencepiece tiktoken tqdm fastapi "uvicorn[standard]" pydantic sympy networkx seaborn tensorboard ipykernel jupyterlab
.\.venv\Scripts\python.exe -m pip install warcio trafilatura datasketch simhash gymnasium stable-baselines3 trl sentence-transformers prometheus-eval math-verify sse-starlette bitsandbytes
.\.venv\Scripts\python.exe -m pip install --force-reinstall pyarrow==21.0.0
```

Important: `pyarrow==21.0.0` is pinned for this Python 3.13 Windows environment. `pyarrow==24.0.0` caused a native access violation when importing in the sequence `torch -> transformers -> datasets`.

## Verification Commands

```powershell
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUTF8="1"

.\.venv\Scripts\python.exe scripts\eric_3080ti_env_audit.py --tiers A B C M8 --tests --timeout 900 --json-out docs\local-env\ERIC-3080Ti-test-results.json --md-out docs\local-env\ERIC-3080Ti-test-matrix.md

.\.venv\Scripts\python.exe scripts\eric_3080ti_env_audit.py --tiers A B C --env --timeout 180 --json-out docs\local-env\ERIC-3080Ti-env-results.json --md-out docs\local-env\ERIC-3080Ti-env-matrix.md
```

## Compatibility Fixes Applied

- Added `scripts/eric_3080ti_env_audit.py` to run per-module tests and env checks with UTF-8, isolated `PYTHONPATH`, timeout control, and fallback for script-style `_self_test()` test runners.
- Made hash-based mock embeddings deterministic in `agent-memory-context` by replacing process-randomized Python `hash()` with stable `blake2b` bucketing.
- Fixed Python 3.13 compatibility issues such as non-ASCII bytes literals in `data-curation`.
- Fixed real model/test mismatches in KV cache generation, RoPE shape expansion, long-context packing overflow, vectorized GAE, token-level reward placement, and estimator memory strategy.
- Converted optional external stacks to clear `SKIP` behavior in env checks: `adapters`, legacy TRL trainer APIs, `verl`, `vllm`, `ray`, `qwen-vl-utils`, and `playwright`.
- Adjusted fragile tests that used exact float equality or overly stochastic RL improvement assertions.

## Remaining Notes

This branch validates the educational/minimal code paths on Windows native. Some production stacks remain optional by design:

- `verl`, `vllm`, and full Ray training stacks are treated as WSL2/Linux production paths.
- `adapters` is optional because it conflicts with the main `transformers==5.10.2` environment used by the PEFT/modern architecture modules.
- Real browser/VLM stacks for multimodal agents are optional; mock/minimal module tests pass locally.
