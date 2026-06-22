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

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）上 V0+V1 验证通过（7/7）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules rlhf-classic
> ```

**Stage 2 — Reward Model**（手写 Bradley-Terry，Anthropic-HH 数据，可用 GPU）：

```powershell
# 真实跑（1k 训练对 / 200 评估对，1 epoch）
python learning/rlhf-classic/src/rm_minimal.py --n-train 1000 --n-eval 200 --batch 4 --epochs 1
# 快速 smoke（验证可跑通）
python learning/rlhf-classic/src/rm_minimal.py --n-train 16 --n-eval 8 --batch 4 --epochs 1 --max-length 128
```

**三段管线各阶段 + 玩具 demo**（无 argparse，直接运行；CPU/小显存即可）：

```powershell
python learning/rlhf-classic/src/sft_minimal.py            # Stage 1 SFT：GPT-2 + prompt-mask NLL（loss 下降）
python learning/rlhf-classic/src/ppo_llm_minimal.py        # Stage 3 PPO：手写 4-model token-level PPO（TinyLM 数值 smoke）
python learning/rlhf-classic/src/reward_hacking_demo.py    # Reward Hacking：长度退化 RM → 检测命中 detected=True
python learning/rlhf-classic/src/cai_minimal.py            # Constitutional AI / RLAIF（mock LLM 演示）
python learning/rlhf-classic/src/capstone_tldr_rlhf.py     # Capstone：TL;DR RLHF 三段管线（SFT+RM+PPO-mock，CPU ~45s）
```

> ✅ **无 trl 依赖**：本模块三段管线（SFT / RM / PPO）**全是手写实现**，不踩 trl 1.5.x 经典 `PPOConfig`/`PPOTrainer` 漂移坑。
> lecture 里的 `sft_trl.py` / `ppo_llm_trl.py` 是"三轨实现"教学示意，文件本身未创建（见上「待继续」），不是可运行入口。
> 真要用 trl 生产路径，参考姊妹模块 `rl-foundations` 的 capstone（带自动回退手写 PPO）。
>
> 📦 **数据集**：RM 用命名空间 id `Anthropic/hh-rlhf`（datasets 5.x 可加载）；加载失败时 `rm_minimal.py` 回退内置 dummy 偏好对（不静默假成功，会打印 WARN）。

**环境自检**：

```powershell
python learning/rlhf-classic/environment/verify_env.py
```

**测试（V2）**：

```powershell
python -m pytest learning/rlhf-classic/src/tests/ -v
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules rlhf-classic --tests
```

