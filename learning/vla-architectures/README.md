# 11.2 vla-architectures — VLA 架构: backbone + 动作头

> **Module 11「具身智能 / VLA / 机器人」· 第 2 专题 (架构)**
> VLA 的标准结构 = **两阶段**: 感知 backbone (真实里 = M10 VLM) + 动作头。本专题用 M10 的 mini-VLM 当 backbone, 组装一个 mini-VLA, 并对比离散 vs 连续动作头。

---

## 这个专题要解决的真问题

- **VLA 长什么样?** → 两阶段: backbone (理解世界, =VLM) + 动作头 (解码动作)。
- **真实 VLA 怎么设计?** → OpenVLA (离散 token) / π (flow-matching) / GR00T (扩散+世界模型); backbone 趋同, **动作头分化**。
- **动作怎么表示?** → 离散 token (复用 LLM/多峰, 但离散化损失) vs 连续 (精度/平滑, 但怕多峰)。
- **怎么接 M10?** → 把 mini-VLM 的 VQA 输出层换成动作头 = mini-VLA (RT-2 做法)。

> **核心信念**: backbone 你 M10 已造好; VLA = 给 VLM 接动作头。mini-VLA → OpenVLA 只差规模。

## 学习路径 (4 讲)
| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-vla-two-stage.md` | VLA = backbone (VLM) + 动作头, 共同骨架 |
| L2 | `lectures/L2-openvla-pi-groot-compare.md` | 三个真实 VLA 对比 + 设计 5 决策 |
| L3 | `lectures/L3-discrete-vs-continuous-actions.md` | 离散 token vs 连续动作的代价 (多峰问题) |
| L4 | `lectures/L4-connecting-m10-backbone.md` | 把 M10 mini-VLM 接动作头 = mini-VLA |

## 动手 (2 个 notebook)
| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-assemble-mini-vla.ipynb` | import M10 mini-VLM 确认是 backbone, 组装 mini-VLA 并 rollout |
| `notebooks/N2-discrete-vs-continuous.ipynb` | 离散 vs 连续动作头: 成功率(都高) + 平滑度(连续 4×) |

## 工具 (`src/`)
- `mini_vla.py` — 两阶段 VLA (backbone + 可换动作头 discrete/continuous) + 训练 + 平滑度评测。**复用 M11.1 toy_env。**

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 全部通过 ✅
```
Python 3.13 / torch (tiny CPU) / numpy / matplotlib。复用 M11.1 toy_env, 离线确定性。

## 完成本专题后你应该能
- [ ] 画出 VLA 两阶段结构 (backbone + 动作头), 说清各自职责
- [ ] 用「5 决策」对比 OpenVLA / π / GR00T
- [ ] 解释离散 vs 连续动作头的代价 (多峰问题为何关键)
- [ ] 说清 mini-VLM (M10) 怎么变 mini-VLA (接动作头)
- [ ] 在 toy 上组装 mini-VLA 并对比两种动作头

---
## 在 Module 11 中的位置
```
  11.1 地基 → 11.2 VLA架构 ◄你在这 → 11.3 扩散动作头 → 11.4 数据/模仿 → 11.5 世界动作模型 → 11.6 sim2real → 11.7 capstone
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module11-embodied-vla-design.md`
