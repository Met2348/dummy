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

## 运行验证（Runbook）

> 上面的 `## 入口` 是【完整版】路线（WSL2 + verl + Ray + vllm + Megatron，真训练 4-6h）。
> 本段是【本地 smoke 版】：`src/` 的 5 个脚本全是**手写数值 GRPO/RLOO/REINFORCE++ demo**，纯 CPU 秒级跑完，
> **不加载模型、不下载数据集、无需 verl/vllm/WSL2**，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）V1 验证通过（**无需改代码**）。
> 这 5 条即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"。一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules reasoning-r1
> ```

每个脚本直接跑（**无需传参，自带 smoke 规模**）：

```powershell
# ⭐ GRPO 核心：group z-score advantage（同 prompt 组内 mean/std 归一化做 baseline，无 critic）+ clip surrogate + unbiased KL
python learning/reasoning-r1/src/grpo_minimal.py
# RLOO：leave-one-out baseline，输出里与 GRPO z-score advantage 同数据并排对比
python learning/reasoning-r1/src/rloo_minimal.py
# REINFORCE++（OpenRLHF）：去 critic + KL 加在 reward（而非 loss）+ 保留 PPO clip
python learning/reasoning-r1/src/reinforce_pp.py
# Capstone Track A 教学轨：Countdown-3 + GRPO（mock 训练 200 步，演示 reward↑/advantage）
python learning/reasoning-r1/src/r1_zero_track_a.py
# Capstone Track B 挑战轨：Qwen-1.5B+4bit LoRA+GSM8K（LoRA setup 伪代码 + aha 词频检测）
python learning/reasoning-r1/src/r1_zero_track_b.py
```

> ⚠️ **mock vs full 说明（诚实标注，非假成功）**：
> - `grpo_minimal` / `rloo_minimal` / `reinforce_pp` 用**随机张量**驱动，演示的是 advantage/loss 的**真实数值算法**（GRPO 输出可见每组 `mean≈0` 证明 z-score baseline 生效），不是 no-op。
> - `r1_zero_track_a` 的训练步是 `mock_train_step`（源码已标注「真训需 actor + tokenizer」），用 `random` 模拟 reward 上升、但**复用真实的 `compute_group_advantage`**；它本就**不**加载 GPT-2-M，因此本地秒级即完成，**无 180s 超时风险**。
> - `r1_zero_track_b` 的 `setup_lora_qwen()` 返回的是**伪代码字符串**（CPU 无法真跑 4bit），但 aha 词频检测是真字符串分析。
> - 完整真训练（GPT-2-M / Qwen-1.5B + GRPO 4-6h）按 `## 入口` 与讲义 13/14 在 5090/24GB + WSL2 跑。
>
> ⚠️ **讲义 flag 漂移**：讲义 13/14 写了 `r1_zero_track_a.py --algo grpo --total_steps 500 --k 8` /
> `r1_zero_track_b.py --base ... --steps 1000`，但这俩脚本**无 argparse**，多余 argv 被**静默忽略**（不报错、仍用硬编码默认）。
> 故本 runbook 用上面的**无 flag** 形态。讲义里 `src/r1_zero_track_a/train_grpo.py`、`src/grpo_verl.py` 是完整版占位路径，本地不存在。

**测试（V2）**：覆盖 RLOO/REINFORCE++ advantage、Countdown/GSM8K reward、format/accuracy reward、aha 词频。

```powershell
python scripts/eric_3080ti_env_audit.py --modules reasoning-r1 --tests
# 或直接：python -m pytest learning/reasoning-r1/src/tests/ -v
```

> 注：本地 smoke 不依赖 `official/repos/DeepSeek-R1` submodule（完整版才用）。

