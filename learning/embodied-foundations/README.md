# 11.1 embodied-foundations — 具身 AI 地基 + tokens-as-actions

> **Module 11「具身智能 / VLA / 机器人」· 第 1 专题 (地基)**
> 把你的 LLM/VLM 能力接到物理世界。核心范式: **tokens-as-actions** —— 把动作当 token, 控制就变成「预测下一个 token」, 复用你 LLM 全套。

---

## 这个专题要解决的真问题

- **具身 AI 是什么?** → 学一个策略 π(动作|观测), 在物理世界**闭环**行动 (多了动作输出+闭环)。
- **为什么用基础模型?** → 重演 NLP 革命: 一个预训练 VLA 吃所有数据、泛化、听语言指令 (胜逐任务硬编码)。
- **怎么复用 LLM?** → **tokens-as-actions**: 动作离散成 token → 控制=next-token 预测。
- **为什么数据是关键?** → 正迁移: 「每条 demo 改进所有机器人」(scaling 福音搬到物理世界)。

> **核心信念**: VLA = M10 (VLM 感知核) + M13 (扩散动作头/世界模型) + 机器人数据。你已懂前两块。

## 学习路径 (4 讲)
| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-what-is-embodied-ai.md` | 具身=学策略π+闭环; 为什么转向机器人基础模型 |
| L2 | `lectures/L2-rt1-rt2-openvla-lineage.md` | RT-1→RT-2→OpenVLA→π/GR00T: 控制的 LLM 化 |
| L3 | `lectures/L3-tokens-as-actions.md` | 动作当 token → 控制=next-token (复用 LLM 全套) |
| L4 | `lectures/L4-data-scaling-logic.md` | 「每条 demo 改进所有机器人」: 数据正迁移/scaling |

## 动手 (2 个 notebook)
| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-env-and-serialize.ipynb` | 跑 2D 到达任务闭环 + 把连续动作离散成 token |
| `notebooks/N2-next-action-prediction.ipynb` | 训「状态→动作 token」预测器, 当策略 rollout (具身版 next-token) |

## 工具 (`src/`)
- `toy_env.py` — 2D 到达任务环境 (状态/动作/转移/专家/成功率)。**M11 全模块共享地基。**
- `action_serialize.py` — 观测/动作 ↔ token (tokens-as-actions 序列化)

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 全部通过 ✅
```
Python 3.13 / torch (tiny CPU) / numpy / matplotlib。玩具 2D 环境离线确定性。

## 完成本专题后你应该能
- [ ] 解释具身 AI 的闭环, 及为什么用基础模型 (vs 逐任务硬编码)
- [ ] 复述 RT-1→RT-2→OpenVLA 谱系, 说清 RT-2 的飞跃
- [ ] 解释 tokens-as-actions 怎么让控制 = next-token 预测
- [ ] 说清数据正迁移「每条 demo 改进所有机器人」的逻辑
- [ ] 在 toy 上训一个 next-action 策略并 rollout

---
## 在 Module 11 中的位置
```
  11.1 地基 ◄你在这 → 11.2 VLA架构 → 11.3 扩散动作头 → 11.4 数据/模仿 → 11.5 世界动作模型 → 11.6 sim2real → 11.7 capstone
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module11-embodied-vla-design.md`
> `toy_env.py` 是 M11 全模块的共享环境。
