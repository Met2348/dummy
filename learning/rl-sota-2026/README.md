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

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）上 V0+V1 验证通过。
> 这些都是**手写 minimal 数值 demo**（无 argparse、纯 CPU、秒级跑完，无需 GPU/大权重/可选栈）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules rl-sota-2026
> ```

**5 个可跑入口**（直跑即出数值，无参数）：

```powershell
# 1. DAPO 4 件套独立实现 + smoke（Clip-Higher / Dynamic Sampling / Token-level PG / Overlong Shaping）
python learning/rl-sota-2026/src/dapo_minimal.py
# 2. Dr.GRPO —— 去掉 GRPO 两个偏置：①advantage 去 std 除法 ②loss 去 1/|o| 长度归一
python learning/rl-sota-2026/src/dr_grpo.py
# 3. VAPO —— Length-Adaptive GAE（λ 随 response 长度动态调整）
python learning/rl-sota-2026/src/vapo_minimal.py
# 4. GenRM —— 生成式 RM（LLM 出 CoT critique 再打分），mock judge 对照 scalar RM
python learning/rl-sota-2026/src/genrm.py
# 5. Capstone —— DAPO 4 件套消融实验（6 配置：baseline + 逐个加 trick + 全开），mock 增量训练
python learning/rl-sota-2026/src/capstone_dapo_ablation.py
```

**DAPO 4 件套机制确在代码里**（非占位 print）：`dapo_minimal.py` 的 `asymmetric_clip_loss`（解耦 ε_low/ε_high）、`is_group_useful` + `dynamic_sampling_rollout`（过滤全对/全错组并真重采）、`token_level_loss` vs `response_level_loss`（聚合粒度不同）、`overlong_shaping`（分段线性 soft penalty）。`capstone_dapo_ablation.py` 逐个开关这 4 个 trick 跑 6 配置并打印 acc/delta 差异。

> ⚠️ **verl 只在讲义**：L01 Slide 14/24 提到 `dapo_verl.py --config configs/dapo_full.yaml` 与 verl recipe，但**本地无该文件**，`src/` 零 verl/vllm/ray/trl import——可跑代码全是手写 minimal（延续本系列 rlhf/dpo/r1 同模式）。runbook 只登记这 5 个手写 demo。
> ⚠️ **Overlong 公式：代码 = 论文，讲义是简化**。L01 Slide 11 写的是 `reward * sigmoid(...)`；`overlong_shaping` 实现的是 DAPO 论文原版的**分段线性 Soft Overlong Punishment**（`penalty=(expected_len-len)/cache_len`，截断超 max 记 -1）。以代码为准。
> ✅ **Dr.GRPO 已对齐论文**（2026-06 修正）：原实现曾是"MAD + 长度惩罚"变体，与 Sea AI Lab 论文取向相反，已重写为忠实版——`dr_grpo_advantage` = `R − mean`（**去掉 std 除法**，消除问题难度偏置）、`grpo_length_weight` vs `dr_grpo_length_weight`（**常数归一替代 1/|o|**，消除响应长度偏置）。直跑可见两组对照：低方差组 GRPO `±1.225` vs Dr.GRPO `±0.05`；长度权重 GRPO `[0.1,0.02,0.005]` vs Dr.GRPO `[0.005,0.005,0.005]`。要点：Dr.GRPO 是"做减法"（去归一化），不是换归一化或加惩罚项。
> ℹ️ **mock 诚实标注**：capstone 的 6 配置 accuracy 是 `mock_ablation_run` 按 trick 叠加的合成值（输出已标 "(mock)"），演示消融**对比结构**而非真实训练；真实增量训练需 5090 24GB。GenRM 同为 mock judge。

**测试（V2）**：

```powershell
python -m pytest learning/rl-sota-2026/src/tests/ -v
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules rl-sota-2026 --tests
```

