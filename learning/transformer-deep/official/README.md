# Official Code: Attention Is All You Need

## Status

- Official / historical source: `tensorflow/tensor2tensor`
- Local path: `learning/transformer-deep/official/repos/tensor2tensor`
- Pinned commit on ERIC-3080Ti: `bafdc1b`
- Reproduction level: `L1 + L4`

This is the historical TensorFlow-era implementation associated with the original Transformer release. It is valuable for reading architecture organization and naming, but it is not the recommended execution path for this Windows workstation. The teaching path should use this repository for source inspection and use the local PyTorch files under `src/` for runnable mechanism reproduction.

## Files To Read First

- `tensor2tensor/models/transformer.py`
- `tensor2tensor/layers/common_attention.py`
- `tensor2tensor/layers/common_layers.py`

Map these files to the local teaching implementation:

- `learning/transformer-deep/src/mha.py`
- `learning/transformer-deep/src/pe_sinusoidal.py`
- `learning/transformer-deep/src/gpt_mini.py`

## Local Setup Notes

Do not install the old Tensor2Tensor stack into the shared `.venv` by default. It can pull old TensorFlow constraints and conflict with the current PyTorch teaching environment.

Recommended local smoke target:

```powershell
.venv\Scripts\python.exe learning\transformer-deep\src\mha.py
```

Expected learning result:

- explain why attention scores have shape `(batch, heads, query_tokens, key_tokens)`;
- explain why scores are divided by `sqrt(d_head)`;
- identify where Q/K/V projection and output projection live in code;
- compare the old official implementation structure with the local minimal PyTorch implementation.

## Notebook

Teaching notebook:

- `learning/transformer-deep/notebooks/01_attention_walkthrough.ipynb`

The notebook should run from the repository root and should not require Tensor2Tensor imports.
