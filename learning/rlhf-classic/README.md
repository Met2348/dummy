# RLHF Classic — InstructGPT 三段管线

> 专题 2 / 7（继承专题 1 PPO 基础）
>
> Design: [2026-06-03-rlhf-classic-design.md](../../docs/superpowers/specs/2026-06-03-rlhf-classic-design.md)
> Plan: [2026-06-03-rlhf-classic.md](../../docs/superpowers/plans/2026-06-03-rlhf-classic.md)

## 概览

12 lecture / 完整三段管线 SFT → RM → PPO / capstone TL;DR 摘要 RLHF / 15h.

## 已完成（本轮）

- L01 InstructGPT 三段管线 lecture
- L03 RM Bradley-Terry lecture
- src/common.py (复用专题 1) + src/rm_minimal.py + tests
- 环境 verify_env.py
- papers 索引

## 待继续

- L02 SFT lecture + src/sft_minimal.py
- L04 PPO for LLM 深化（基本继承专题 1）
- L05 RLHF 工程细节
- L06 RLAIF / Constitutional AI
- L07 LLaMA-2 RLHF (rejection sampling)
- L08 Sparrow
- L09 Reward Hacking
- L10 RLHF Pitfalls
- L11 多目标 RLHF（Safe / MaxMin）
- L12 Capstone TL;DR 摘要 RLHF

## 入口

```bash
python learning/rlhf-classic/environment/verify_env.py
python learning/rlhf-classic/src/rm_minimal.py
pytest learning/rlhf-classic/src/tests/
```

