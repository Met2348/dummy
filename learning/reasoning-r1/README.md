# Reasoning R1 ⭐⭐⭐⭐⭐ — 系列高峰

> 专题 5 / 7 — **整个 RL 系列的高峰**。完整覆盖 GRPO / R1 / R1-Zero / Kimi k1.5 / TinyZero / Open-R1。
>
> Design: [link](../../docs/superpowers/specs/2026-06-03-reasoning-r1-design.md)
> Plan: [link](../../docs/superpowers/plans/2026-06-03-reasoning-r1.md)

⚠️ **本专题切换到 WSL2**：verl + Ray + vllm + Megatron。

## 已完成（本轮）

- L00 WSL2 + verl + vllm 配置 lecture
- L02 GRPO 完整推导 lecture (28 slides)
- src/grpo_minimal.py: 手写 GRPO 核心组件
- src/rewards/{format,accuracy}_reward.py: R1-Zero reward 函数
- src/tests/test_format_accuracy_reward.py: 严格单元测试
- environment + papers + README

## Capstone 双轨

- **Track A (必跑)**: GPT-2-M (355M) + Countdown-3
  - 算法：REINFORCE / RLOO / GRPO / GRPO+Clip-Higher 4 算法对照
  - format reward 5%→95%, accuracy 5%→15%
  - 5090 24GB, 总 ~6h
- **Track B (选跑)**: Qwen-1.5B + 4bit LoRA + GSM8K-tiny
  - 算法：GRPO + DAPO
  - 期望 aha moment 词频 ≥ 5%
  - 5090 24GB, 单跑 ~4h

## 待续

L01 o1 / L03 R1-Zero / L04 R1 / L05 Kimi / L06-L09 RLOO/ReMax/VinePPO/REINFORCE++
L10 TinyZero / L11 Open-R1 / L12 Spurious Rewards ⚠️
L13-L14 双轨 Capstone / L15 总结

## 入口

```bash
# WSL2 内
cd /mnt/c/Workspace/dummy/learning/reasoning-r1
python environment/verify_env.py
python src/grpo_minimal.py
pytest src/tests/
```

