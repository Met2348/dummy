# 11.7 embodied-graduation — Module 11 Capstone

> **Module 11「具身智能 / VLA / 机器人」· 第 7 专题 (毕业设计)** 🎓
> 把 M11 全栈 (tokens-as-actions→VLA→扩散动作头→数据→世界模型→sim2real) 装配成一个栈, 跑 mini-benchmark, 并用 M9 找 gap 框架产出研究 idea 卡。从「会用」到「能推进」。

---

## 这个专题要做的两件事

1. **装配全栈 + mini-benchmark**: 一次 import M11 全 7 个 src, 多方法 (BC / mini-VLA / 世界模型MPC) 在同一 toy 任务比成功率 (LIBERO/CALVIN 思路)。
2. **找研究 gap**: 用 M9 框架在具身全栈挖 6 个 gap, 产 idea 卡, 挑最匹配你 (NLP + IsaacLab) 的。

## 学习路径 (4 讲)
| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-the-embodied-stack.md` | M11 全栈回顾 + 三条核心信念 |
| L2 | `lectures/L2-assemble-end-to-end-vla.md` | 拆 OpenVLA/π/GR00T = M11/M10/M13 部件 + 7问 |
| L3 | `lectures/L3-embodied-eval-and-gaps.md` | 具身评测 (LIBERO/CALVIN) + 找 gap |
| L4 | `lectures/L4-graduation-and-roadmap.md` | M11 在 2026, 连 M13/M12, 你的路线 |

## 动手 (2 个 notebook)
| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-assemble-and-benchmark.ipynb` | 装配检查 (全7 src) + mini-benchmark (多方法成功率) |
| `notebooks/N2-research-idea-cards.ipynb` | 产 6 张具身 idea 卡, 高亮最匹配你的 (数据/sim2real) |

## 工具 (`src/`)
- `embodied_capstone.py` — 跨专题装配检查 (import M11 全栈) + 研究 gap 雷达 + idea 卡生成器

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 全部通过 ✅ (含 M11 全栈装配)
```
Python 3.13 / torch (tiny CPU) / numpy / matplotlib。复用 M11 全栈 toy, 离线确定性。

## 完成本专题后你应该能
- [ ] 一张图串起 M11 全栈, 说清三条核心信念
- [ ] 把 OpenVLA/π/GR00T 拆成 M11/M10/M13 部件 (7问)
- [ ] 说清具身评测思路 (LIBERO/CALVIN) 与坑
- [ ] 在具身全栈找 gap, 写合格 idea 卡 (含最小可验证实验)
- [ ] 挑一个最匹配你的 gap (数据高效 / sim2real), 细化成研究种子

---
## 在 Module 11 中的位置
```
  11.1→11.2→11.3→11.4→11.5→11.6 → 11.7 capstone ◄你在这 (Module 11 毕业 🎓)
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module11-embodied-vla-design.md`
> 下一站: M12 机制可解释性 (打开你造的 VLA/LLM 看内部)。
