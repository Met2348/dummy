# Official Code: FlashAttention

## Status

- Official source: `Dao-AILab/flash-attention`
- Local path: `learning/kernel-engineering/official/repos/flash-attention`
- Pinned commit on ERIC-3080Ti: `bc58abc`
- Reproduction level: `L1 + optional L3`

This is the official FlashAttention implementation. It contains CUDA/C++ kernels, Hopper-specific paths, Python interfaces, tests, and benchmarks. On this Windows workstation, the default teaching path is source inspection plus local CPU-safe mechanism reproduction. Building the official package should be treated as a later WSL2/Linux CUDA task.

## Files To Read First

- `README.md`: supported GPUs, installation notes, and API overview.
- `flash_attn/flash_attn_interface.py`: Python-facing interface.
- `csrc/flash_attn/flash_api.cpp`: native binding entrypoint.
- `hopper/softmax.h`: hardware-aware softmax details for newer kernels.
- `hopper/tile_size.h`: tile-size choices and scheduling context.

Map these files to the local teaching implementation:

- `learning/kernel-engineering/src/flash_attention.py`
- `learning/kernel-engineering/src/triton_style.py`
- `learning/kernel-engineering/src/capstone_attn_speedup.py`
- `learning/kernel-engineering/src/tests/test_all.py`

## Local Setup Notes

Do not install or build `flash-attn` in the shared `.venv` by default. The official package may need a Linux CUDA toolchain, compatible PyTorch, ninja, and GPU architecture-specific compilation.

Recommended smoke target:

```powershell
.venv\Scripts\python.exe learning\kernel-engineering\src\tests\test_all.py
```

Expected learning result:

- explain online softmax with running max `m` and normalizer `l`;
- explain why full `N x N` attention materialization is the memory bottleneck;
- compare naive attention and block-wise FlashAttention outputs;
- know where official CUDA kernels and Python APIs live.

## Notebook

Teaching notebook:

- `learning/kernel-engineering/notebooks/01_flashattention_walkthrough.ipynb`

The notebook should run from the repository root and should not require compiling official CUDA kernels.
