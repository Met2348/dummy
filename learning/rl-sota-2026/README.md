# RL SOTA 2025-2026 — 算法升级清单

> 专题 6 / 7 — DAPO / VAPO / PRIME / Skywork-Reward V2 / Self-Taught Evaluator
>
> Design: [link](../../docs/superpowers/specs/2026-06-03-rl-sota-2026-design.md)
> Plan: [link](../../docs/superpowers/plans/2026-06-03-rl-sota-2026.md)

## 已完成（本轮）

- L01 DAPO 4 件套 lecture (32 slides) ⭐⭐⭐⭐⭐
- src/dapo_minimal.py: 4 件套独立实现 + 测试演示
- src/tests/test_dapo_each_trick.py: 每个 trick 独立单测
- environment / papers / README

## 待续

L02-L11 (VAPO/Dr.GRPO/GSPO/PRIME-full/GenRM/Skywork V2/Self-Taught/JudgeBench/Nash-MD/Scaling)
L12 Capstone: DAPO 4 件套消融实验

## Capstone

基于专题 5 capstone-A 训出的 R1-Zero baseline ckpt 增量训练:
- 6 配置 (baseline + 4 单独 trick + 全开)
- 每 200 step
- 5090 24GB ~ 6h 总

## 入口

```bash
python learning/rl-sota-2026/src/dapo_minimal.py    # smoke
pytest learning/rl-sota-2026/src/tests/             # 4 件套测试
```

