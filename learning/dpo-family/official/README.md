# Official Code: Direct Preference Optimization

## Status

- Official source: `eric-mitchell/direct-preference-optimization`
- Local path: `learning/dpo-family/official/repos/direct-preference-optimization`
- Pinned commit on ERIC-3080Ti: `f8b8c0f`
- Reproduction level: `L2`

This is the author implementation for Direct Preference Optimization. It is appropriate for source inspection and small smoke tests, but full preference fine-tuning still requires model downloads, datasets, and GPU time.

## Files To Read First

- `trainers.py`: loss computation and trainer structure.
- `preference_datasets.py`: preference pair formatting.
- `config/loss/dpo.yaml`: DPO loss settings.
- `config/model/*.yaml`: model-specific training configs.

Map these files to the local teaching implementation:

- `learning/dpo-family/src/dpo_minimal.py`
- `learning/dpo-family/src/capstone_dpo_comparison.py`
- `learning/dpo-family/src/tests/test_dpo_loss_equivalence.py`

## Local Setup Notes

Do not install the full official requirements into the shared `.venv` before checking conflicts. The first local target is the tensor-level DPO loss, which only needs PyTorch.

Recommended smoke target:

```powershell
.venv\Scripts\python.exe -m pytest learning\dpo-family\src\tests\test_dpo_loss_equivalence.py -q
```

Expected learning result:

- describe a preference batch as `(prompt, chosen, rejected)`;
- compute actor/reference log-probability differences;
- explain why DPO optimizes a logistic classification objective over pairwise margins;
- understand how `beta` changes the strength of the preference update.

## Notebook

Teaching notebook:

- `learning/dpo-family/notebooks/01_dpo_walkthrough.ipynb`

The notebook should run from the repository root and should not require model downloads.
