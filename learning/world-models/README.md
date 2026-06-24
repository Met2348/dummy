# 13.5 world-models — 世界模型: 学环境动态, 脑内想象

> **Module 13「扩散/生成式媒体/世界模型」· 第 5 专题 (生成→预测→决策的枢纽)**
> M13.4 的视频生成加上「动作条件」就成了**世界模型**: 学环境怎么从一个状态走到下一个。学会后能在脑内**想象** rollout, 不碰真环境就规划。这是连接生成 (M13) 与具身 (M11) 的共享地基。

---

## 这个专题要解决的真问题

- **什么是世界模型?** → 学转移动态 $P(s_{t+1}\mid s_t,a_t)$: 给状态+动作, 预测下一状态。
- **学会能干嘛?** → **想象 rollout** (脑内链式预测), 用于 model-based 规划/省真实交互/机器人预演。
- **视频模型是世界模型吗?** → 是隐式的 (L2, 2024 洞察): 为生成逼真视频被迫学物理; 但物理仍是近似。
- **核心难题?** → **多步误差累积**: 单步准、多步发散 (同长程视频不一致)。
- **怎么接具身?** → 世界模型 = 机器人的想象空间; `world_model.py` 与 M11 同源共享。

## 学习路径 (4 讲)
| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-what-is-a-world-model.md` | 世界模型=学转移; imagine rollout; 预测Δ更稳; 误差累积 |
| L2 | `lectures/L2-video-models-as-simulators.md` | 视频生成模型是涌现的隐式世界模型 (2024 洞察, 批判看) |
| L3 | `lectures/L3-interactive-playable-worlds.md` | 加动作条件→可玩世界 (Genie/Vid2World/AVID) |
| L4 | `lectures/L4-world-models-and-embodiment.md` | 世界模型×具身: model-based 决策, 通往 M11 |

## 动手 (2 个 notebook)
| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-learn-world-model.ipynb` | 玩具 2D 环境训世界模型, 给动作让它**想象**轨迹, 和真环境对照 |
| `notebooks/N2-prediction-quality.ipynb` | 量化**多步误差累积**: 单步准、多步发散 (世界模型核心难题) |

## 工具 (`src/`)
- `world_model.py` — 动作条件世界模型 (state+action→Δstate) + 想象 rollout + 多步误差评测 (torch CPU)。**与 M11 world-action-models 同源共享。**

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 全部通过 ✅
```
Python 3.13 / torch (tiny CPU) / numpy / matplotlib。玩具 2D 环境离线确定性。

## 完成本专题后你应该能
- [ ] 解释世界模型 = 学转移动态, 以及 imagine rollout 的用途
- [ ] 说清「视频模型是隐式世界模型」的论点与批判 (物理是统计近似)
- [ ] 解释怎么加动作条件做可玩世界 (用 M13.3 条件注入, 无新机制)
- [ ] 量化并解释多步误差累积, 及对抗手段 (MPC/重规划)
- [ ] 画出世界模型如何连接生成 (M13) 与具身 (M11)

---
## 在 Module 13 中的位置
```
  13.1→13.2→13.3→13.4 视频 → 13.5 世界模型 ◄你在这 → 13.6 dLLM → 13.7 capstone
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module13-diffusion-world-models-design.md`
> `world_model.py` 是 M11 具身模块的共享地基。
