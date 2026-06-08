# Official Artifact: InstructGPT / Following Instructions With Human Feedback

## Status

- Official source: `openai/following-instructions-human-feedback`
- Local path: `learning/rlhf-classic/official/repos/following-instructions-human-feedback`
- Pinned commit on ERIC-3080Ti: `5c0534c`
- Reproduction level: `L1 + L4`

This OpenAI repository is an official artifact for the InstructGPT paper family, but it is not a full training implementation. It contains a README, model card, and automatic-evaluation sample outputs. Full SFT, reward-model, and PPO training at InstructGPT scale is not reproducible from this artifact on a single workstation.

## Files To Read First

- `README.md`: paper and model-release context.
- `model-card.md`: intended use, risks, and limitations.
- `automatic-eval-samples/*.csv`: examples of automatic evaluation outputs.

Map these official artifacts to the local teaching implementation:

- `learning/rlhf-classic/src/sft_minimal.py`
- `learning/rlhf-classic/src/rm_minimal.py`
- `learning/rlhf-classic/src/ppo_llm_minimal.py`
- `learning/rlhf-classic/src/tests/test_three_stage_pipeline.py`

## Local Setup Notes

Use the official repository for context and evaluation examples. Use local toy code to learn the actual mechanics:

```text
SFT: prompt -> demonstration -> supervised next-token loss
RM: (chosen, rejected) -> Bradley-Terry pairwise loss
PPO: actor/ref/RM/critic -> KL-regularized policy update
```

Recommended smoke target:

```powershell
.venv\Scripts\python.exe -m pytest learning\rlhf-classic\src\tests\test_three_stage_pipeline.py -q
```

Expected learning result:

- explain the three-stage RLHF pipeline;
- derive Bradley-Terry reward-model loss;
- describe why PPO uses a frozen reference model and KL penalty;
- say clearly what the official artifact does and does not let us reproduce.

## Notebook

Teaching notebook:

- `learning/rlhf-classic/notebooks/01_instructgpt_walkthrough.ipynb`

The notebook should use tiny synthetic tensors and should not require model downloads.
