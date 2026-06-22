# Process Reward & Verification — 推理 RL 工具箱

> 专题 4 / 7 — 12 lecture / PRM 训练 + BoN + MCTS + LLM-as-Judge
>
> Design: [link](../../docs/superpowers/specs/2026-06-03-process-reward-design.md)
> Plan: [link](../../docs/superpowers/plans/2026-06-03-process-reward.md)

## 概览

| L | 主题 | 一句话 |
|---|------|--------|
| 01 | ORM vs PRM | outcome / process 监督 |
| 02 | Let's Verify (Lightman 2023) | OpenAI 首个 large-scale PRM |
| 03 | PRM 训练实战 | step 划分 + soft label |
| 04 | Math-Shepherd | MC rollout 自动生成 PRM 数据 |
| 05 | PPM | preference PRM |
| 06 | PRIME | 隐式 PRM |
| 07 | RLVR | verifiable rewards |
| 08 | Tree Search | BoN / Beam / ToT |
| 09 | MCTS for LLM | UCT / MCTS-DPO |
| 10 | LLM-as-Judge | G-Eval / Prometheus 2 |
| 11 | RM Pitfalls | length / sycophancy / position |
| 12 | Capstone | GSM8K PRM + BoN-32 |

## 运行验证（Runbook）

> 本模块的"可运行入口"即 [`runbook.yaml`](runbook.yaml) 登记的 7 个 PRM 工具箱直跑 demo，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）V1 验证通过（**无需改代码**）。
> 一键复验：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules process-reward
> ```

7 个脚本均**无需传参**（自带 smoke 规模，纯数值/搜索 demo，CPU 即可，秒级跑完）：

```powershell
# PRM 步级打分头 + 3-way loss + 4 种聚合（Lightman 2023）
python learning/process-reward/src/prm_minimal.py
# Best-of-N + majority / weighted-BoN 重排策略对比
python learning/process-reward/src/bon_search.py
# Math-Shepherd：MC-rollout 自动生成 PRM 步级 label（Wang 2024）
python learning/process-reward/src/math_shepherd_data_gen.py
# MCTS for LLM：UCT 选择/扩展/模拟/回传（rStar-Math 风格 toy）
python learning/process-reward/src/mcts_llm.py
# PRIME：隐式 PRM，actor/ref 对数差 → 步级 reward（Tsinghua 2025）
python learning/process-reward/src/prime_minimal.py
# RLVR：可验证奖励（GSM8K / Countdown / Code / Format / Combined）
python learning/process-reward/src/rlvr_demo.py
# Capstone：GSM8K PRM 引导 Best-of-N（mock 候选，100 题 × 32 路真实重排）
python learning/process-reward/src/capstone_prm_bon.py
```

> 注（demo 性质，非 bug）：
> - **Capstone 用 mock 候选生成器**（`mock_generate` 随机模拟 Qwen-0.5B 的 32 路采样 + PRM 分），但**重排对比是真的**——它跑 100 题 × 32 候选的 greedy/majority/BoN/weighted-BoN 全流程并打印准确率。要换真模型采样，把 `mock_generate` 接到 Qwen2.5-0.5B + 真 PRM 即可。其 mock 中"正确答案恒高分"使 BoN ≈ 完美 oracle（准确率打满），仅作管线演示。
> - **Math-Shepherd / MCTS** 用 toy `rollout_fn` / `expand_fn`（`random` 模拟步级成功率），演示自动打标与 UCT 搜索逻辑本身，非接真 LM。
> - 这些脚本无 argparse → runbook 内标 `v0: false`（不跑 `--help` 探针，直接 smoke 直跑到完成）。

**测试（V2）**：13 个单测覆盖 PRM 聚合 / BoN / RLVR 规则奖励 / PRIME shape / Math-Shepherd 边界：

```powershell
python scripts/eric_3080ti_env_audit.py --modules process-reward --tests
# 或单独：python -m pytest learning/process-reward/src/tests/ -v
```

**环境自检（--env）**：`environment/verify_env.py` 检查 math-verify/sympy/networkx + 试加载 GSM8K（缺依赖/离线只 WARN，不阻塞 demo）。
