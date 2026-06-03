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

## 已完成（本轮）

- environment scaffold
- papers 索引
- README 框架

## 入口

待 lecture/src 填充后跑 `python environment/verify_env.py`
