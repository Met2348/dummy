# 11.5 world-action-models — 具身世界模型 + 规划

> **Module 11「具身智能 / VLA / 机器人」· 第 5 专题 (从模仿到想象)**
> 从纯模仿 (M11.4 BC) 到带想象: **世界模型**学转移动态, 学会后能想象 + 规划 (MPC), 且**数据可随机探索 (零专家)**。接你 M13.5 的 `world_model.py` (同源)。

---

## 这个专题要解决的真问题

- **怎么不靠专家学控制?** → **世界模型**: 从随机探索学 (state,action)→next, 再 MPC 规划。
- **怎么用世界模型?** → 想象 rollout + 规划 (MPC, 短视野+重规划对抗误差累积)。
- **训练哲学?** → **先学想象 (预训练世界模型) + 再学行动** (NVIDIA 路线, 同 LLM 预训练+微调)。
- **省交互?** → **model-based RL**: 在想象里 rollout 训策略 (Dreamer), 样本效率暴涨。
- **升到图像?** → **视频模型 = 图像世界模型** (M13.4/13.5)。

> **核心信念**: M11.5 = M13 世界模型嫁接机器人决策; `world_model.py` 与 M13.5 同源 (一面生成一面决策)。

## 学习路径 (4 讲)
| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-world-model-for-control.md` | 世界模型学转移→想象→MPC 规划 (零专家) |
| L2 | `lectures/L2-imagine-then-act.md` | 先学想象再学行动 (预训练世界模型) |
| L3 | `lectures/L3-model-based-rl.md` | model-based RL: 想象里练策略, 省交互 |
| L4 | `lectures/L4-video-models-as-simulators.md` | 视频模型 = 图像世界模型 (接 M13) |

## 动手 (2 个 notebook)
| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-world-model-planning.ipynb` | 随机数据学世界模型 + MPC 规划到目标 (**零专家**) |
| `notebooks/N2-model-based-vs-free.ipynb` | model-based(随机) vs model-free/BC(专家) 样本效率 |

## 工具 (`src/`)
- `world_model.py` — 具身世界模型 (state,action→Δ) + 想象 + MPC 规划。**与 M13.5 同源, 复用 M11.1 toy_env。**

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 全部通过 ✅
```
Python 3.13 / torch (tiny CPU) / numpy / matplotlib。复用 M11.1 toy_env, 离线确定性。

## 完成本专题后你应该能
- [ ] 解释世界模型 (学转移) + 想象 + MPC 规划, 及它为何不需专家
- [ ] 说清「先学想象, 再学行动」训练哲学 (同 LLM 预训练+微调)
- [ ] 解释 model-based RL 怎么省交互 (想象里 rollout)
- [ ] 对比 model-based vs model-free 的数据来源与样本效率
- [ ] 说清视频模型 = 图像世界模型 (M13→M11 桥)

---
## 在 Module 11 中的位置
```
  11.1 → 11.2 → 11.3 → 11.4 → 11.5 世界动作模型 ◄你在这 → 11.6 sim2real → 11.7 capstone
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module11-embodied-vla-design.md`
> `world_model.py` 与 M13.5 同源 (世界模型概念的生成侧/决策侧两面)。
