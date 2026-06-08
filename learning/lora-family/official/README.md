# Official Code: LoRA

## Status

- Official source: `microsoft/LoRA`
- Local path: `learning/lora-family/official/repos/LoRA`
- Pinned commit on ERIC-3080Ti: `c4593f0`
- Reproduction level: `L2 + optional L3`

This is the Microsoft LoRA implementation. The most useful teaching artifact is `loralib`, especially the layer wrappers that freeze a base weight and add trainable low-rank matrices.

## Files To Read First

- `loralib/layers.py`: LoRA layer definitions.
- `loralib/utils.py`: helper functions for marking LoRA parameters trainable and saving LoRA state.
- `README.md`: usage examples and paper context.

Map these files to the local teaching implementation:

- `learning/lora-family/src/lora_minimal.py`
- `learning/lora-family/src/lora_peft.py`
- `learning/lora-family/src/tests/test_lora_consistency.py`

## Local Setup Notes

Do not run the large NLU examples first. Begin with matrix-level behavior:

```text
base output = x W0^T
LoRA output = x W0^T + alpha / r * x A^T B^T
delta W = alpha / r * B A
```

Recommended smoke target:

```powershell
.venv\Scripts\python.exe -m pytest learning\lora-family\src\tests\test_lora_consistency.py -q
```

Expected learning result:

- explain why LoRA has `r * d_in + d_out * r` trainable parameters;
- show that zero-initialized `B` makes the initial LoRA branch a no-op;
- merge LoRA weights into the base weight for inference;
- compare official `loralib.Linear` with the local `LoRALinear`.

## Notebook

Teaching notebook:

- `learning/lora-family/notebooks/01_lora_walkthrough.ipynb`

The notebook should run from the repository root and should not download GPT-2 or GLUE data.
