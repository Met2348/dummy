# Official Code: Proximal Policy Optimization

## Status

- Official source: `openai/baselines`
- Local path: `learning/rl-foundations/official/repos/baselines`
- Pinned commit on ERIC-3080Ti: `ea25b9e`
- Reproduction level: `L2`

OpenAI Baselines contains historical PPO implementations. The code is useful for reading the real training loop, rollout runner, value function loss, entropy bonus, and clipping behavior. It is old TensorFlow-era code, so the local PyTorch implementation remains the primary runnable teaching path.

## Files To Read First

- `baselines/ppo2/ppo2.py`: PPO2 learning loop.
- `baselines/ppo2/model.py`: policy/value model and loss setup.
- `baselines/ppo2/runner.py`: rollout collection.
- `baselines/ppo1/pposgd_simple.py`: earlier PPO style.

Map these files to the local teaching implementation:

- `learning/rl-foundations/src/ppo_minimal.py`
- `learning/rl-foundations/src/gae.py`
- `learning/rl-foundations/src/tests/test_ppo_consistency.py`

## Local Setup Notes

Do not install old Baselines dependencies into the shared `.venv` by default. Use it for source inspection and use local PyTorch tests for executable learning.

Recommended smoke target:

```powershell
.venv\Scripts\python.exe -m pytest learning\rl-foundations\src\tests\test_ppo_consistency.py -q
```

Expected learning result:

- explain the clipped surrogate for positive and negative advantages;
- compute GAE by hand for a short rollout;
- explain why PPO reuses rollout data for multiple epochs;
- identify where runner, model, and loss live in the official code.

## Notebook

Teaching notebook:

- `learning/rl-foundations/notebooks/01_ppo_walkthrough.ipynb`

The notebook should run from the repository root and should not require Gym environment training.
