# Official Artifact: DeepSeek-R1

## Status

- Official source: `deepseek-ai/DeepSeek-R1`
- Local path: `learning/reasoning-r1/official/repos/DeepSeek-R1`
- Pinned commit on ERIC-3080Ti: `0cf7856`
- Reproduction level: `L1 + L4`

The official repository contains the DeepSeek-R1 report artifact, README, license, and benchmark figure. It does not provide the full large-scale RL training system. Use it for source context and model-family documentation; use local tensor-level code to learn GRPO, rule rewards, and reasoning-RL mechanics.

## Files To Read First

- `README.md`: model family, usage notes, and result summary.
- `DeepSeek_R1.pdf`: official report copy.
- `figures/benchmark.jpg`: benchmark overview.

Map these files to the local teaching implementation:

- `learning/reasoning-r1/src/grpo_minimal.py`
- `learning/reasoning-r1/src/r1_zero_track_a.py`
- `learning/reasoning-r1/src/r1_zero_track_b.py`
- `learning/reasoning-r1/src/rewards/format_reward.py`
- `learning/reasoning-r1/src/rewards/accuracy_reward.py`

## Local Setup Notes

The honest local target is mechanism reproduction:

```text
prompt -> k sampled responses -> rule rewards -> group-normalized advantages -> GRPO-style clipped loss
```

Recommended smoke targets:

```powershell
.venv\Scripts\python.exe learning\reasoning-r1\src\grpo_minimal.py
.venv\Scripts\python.exe -m pytest learning\reasoning-r1\src\tests -q
```

Expected learning result:

- compute group-relative advantages by hand;
- explain why GRPO removes a learned critic in this teaching view;
- distinguish official model/report artifacts from reproducible training code;
- map format and accuracy rewards to local safe toy tasks.

## Notebook

Teaching notebook:

- `learning/reasoning-r1/notebooks/01_deepseek_r1_walkthrough.ipynb`

The notebook should run from the repository root and should not require model downloads.
