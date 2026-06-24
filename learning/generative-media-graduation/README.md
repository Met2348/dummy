# 13.7 generative-media-graduation — Module 13 Capstone

> **Module 13「扩散/生成式媒体/世界模型」· 第 7 专题 (毕业设计)** 🎓
> 把 M13 全链 (扩散→流→DiT→视频→世界模型→dLLM) 装配成一个栈, 并用 M9 的找 gap 框架产出研究 idea 卡。从「会用」走到「能推进」。

---

## 这个专题要做的两件事

1. **装配全链**: 一次 import M13 全 6 专题的 src, 各跑最小烟测 + 统一生成画廊 (一口气驱动扩散/流/DiT/dLLM) —— 证明你掌握的是完整生成式媒体栈, 不是碎片。
2. **找研究 gap**: 用 M9 框架在 M13 全链挖 7 个研究 gap, 产出 idea 卡 (问题/为什么难/最小可验证实验/连接), 挑最匹配你 (NLP 背景) 的。

## 学习路径 (4 讲)
| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-the-whole-stack.md` | M13 全链回顾 + 三条核心信念 (生成=预测/部件组合+规模/架构=成本优化) |
| L2 | `lectures/L2-assembling-real-systems.md` | 拆解 SD/Sora/世界模型/dLLM = 哪些 M10/M13 部件 + 7问拆解法 |
| L3 | `lectures/L3-finding-research-gaps.md` | 用 M9 框架挖 M13 研究 gap, idea 卡四要素 |
| L4 | `lectures/L4-graduation-and-roadmap.md` | M13 在 2026 的位置, 连 M11/M12, 给你的下一步路线 |

## 动手 (2 个 notebook)
| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-assemble-and-generate.ipynb` | 装配检查 (全 6 专题 src) + 统一生成画廊 (扩散/流/DiT/dLLM 一屏) |
| `notebooks/N2-research-idea-cards.ipynb` | 产出 7 张研究 idea 卡, 高亮最匹配你的 dLLM 相关 gap |

## 工具 (`src/`)
- `generative_capstone.py` — 跨专题装配检查 (import M13 全链 src + 烟测) + 研究 gap 雷达 + idea 卡生成器

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 全部通过 ✅ (含 M13 全 6 专题装配)
```
Python 3.13 / torch (tiny CPU) / numpy / matplotlib。复用 M13 全链 toy, 离线确定性。

## 完成本专题后你应该能
- [ ] 一张图串起 M13 全链, 说清三条核心信念
- [ ] 把 SD/Sora/世界模型/dLLM 拆成 M10/M13 部件 (7问拆解法)
- [ ] 在 M13 全链系统地找研究 gap, 写出合格 idea 卡 (含最小可验证实验)
- [ ] 说清 M13 怎么连 M11 (共享世界模型) / M12 (打开生成模型)
- [ ] 挑一个最匹配自己的 gap, 细化成研究种子

---
## 在 Module 13 中的位置
```
  13.1→13.2→13.3→13.4→13.5→13.6 → 13.7 capstone ◄你在这 (Module 13 毕业 🎓)
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module13-diffusion-world-models-design.md`
> 下一站: M11 具身/VLA (共享 `world_model.py`) 或 M12 机制可解释性。
