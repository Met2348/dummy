# 11.4 robot-data-imitation — 模仿学习 + 数据 scaling

> **Module 11「具身智能 / VLA / 机器人」· 第 4 专题 (数据)**
> 动作头怎么训? **行为克隆 (BC)**: 从专家 demo 监督学。本专题讲 BC 原理 + 经典毛病 (分布漂移) + 机器人界的数据 scaling 教训 + 人类视频当桥。

---

## 这个专题要解决的真问题

- **怎么训动作头?** → **BC**: state→action 监督学专家 demo (你最熟的范式)。
- **BC 的坑?** → **分布漂移/复合误差**: 偏出专家分布就滚雪球 (同 M13.5 误差累积)。
- **数据从哪来?** → 人类**遥操作** (贵/慢/要人); 质量决定 BC 上限。
- **怎么变好?** → **数据 scaling** (更多更广 demo) + 与 VLM 数据 **co-train** 防遗忘。
- **怎么降本?** → **人类 egocentric 视频** 当桥 (embodiment gap 是难点)。

## 学习路径 (4 讲)
| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-behavior-cloning.md` | BC = 监督学专家; 分布漂移/复合误差 |
| L2 | `lectures/L2-teleop-data-quality.md` | 遥操作采集 (贵); 数据质量决定 BC 上限 |
| L3 | `lectures/L3-data-scaling-cotraining.md` | 数据量→成功率; 与 VLM co-train 防遗忘 |
| L4 | `lectures/L4-egocentric-video-bridge.md` | 人类视频当数据桥 (embodiment gap) |

## 动手 (2 个 notebook)
| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-behavior-cloning.ipynb` | 在 toy 上 BC + 看 demo 太少时的分布漂移 |
| `notebooks/N2-data-scaling.ipynb` | 画数据 scaling 曲线 (demo 数量 vs 成功率), 找够用拐点 |

## 工具 (`src/`)
- `bc_train.py` — 行为克隆 (state→action 回归) + 数据 scaling 曲线。**复用 M11.1 toy_env。**

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 全部通过 ✅
```
Python 3.13 / torch (tiny CPU) / numpy / matplotlib。复用 M11.1 toy_env, 离线确定性。

## 完成本专题后你应该能
- [ ] 解释 BC = 监督学专家, 及分布漂移/复合误差
- [ ] 说清遥操作采集的成本与数据质量为何决定 BC 上限
- [ ] 画数据 scaling 曲线并干净地做实验 (单变量, 接 9.4)
- [ ] 解释与 VLM 数据 co-train 为何防遗忘 (接 M10.3)
- [ ] 说清人类视频当数据桥的诱惑与 embodiment gap

---
## 在 Module 11 中的位置
```
  11.1 → 11.2 → 11.3 → 11.4 数据/模仿 ◄你在这 → 11.5 世界动作模型 → 11.6 sim2real → 11.7 capstone
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module11-embodied-vla-design.md`
