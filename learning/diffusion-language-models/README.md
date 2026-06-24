# 13.6 diffusion-language-models — dLLM: 文本也能扩散

> **Module 13「扩散/生成式媒体/世界模型」· 第 6 专题 (绕一圈回 NLP 本行)**
> 主流 LLM 是自回归 (AR, 左到右一次一个 token)。dLLM (如 LLaDA) 改用**离散/masked 扩散**: 从全 [MASK] 出发**并行迭代解码**。这是 2025-2026 的范式迁移热点, 直接关你 NLP。

---

## 这个专题要解决的真问题

- **文本怎么扩散?** → **masked diffusion**: 前向逐步把 token 换 [MASK], 模型学填被遮位。
- **怎么生成?** → 从全 [MASK] **按置信度并行迭代解码** (非自回归, T 轮可 < L)。
- **比 AR 强在哪?** → **双向 infilling / 可控编辑**: 看左右全文, 天生会"完形填空"。
- **代价是什么?** → 并行度 vs 质量旋钮 (轮数); 生态远不如 AR 成熟 (open)。
- **为什么 NLP 人该关注?** → AR 之外的另一条生成范式, 并行解码 + 双向有结构性优势。

> **核心对照**: 机制都是 transformer, 只换「生成范式」—— AR 因果单向左到右; dLLM 双向任意位并行。

## 学习路径 (4 讲)
| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-text-can-diffuse.md` | masked diffusion: 文本扩散 = 逐步遮盖+学填, 并行解码 |
| L2 | `lectures/L2-ar-vs-diffusion-lm.md` | AR vs dLLM 机制差异 (因果单向 vs 双向并行) |
| L3 | `lectures/L3-why-nlp-should-care.md` | 为什么值得关注: 并行解码/可控生成/双向 infilling |
| L4 | `lectures/L4-dllm-state-and-gaps.md` | dLLM 现状与开放问题 (用 M9.3 批判读) |

## 动手 (2 个 notebook)
| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-masked-diffusion-lm.ipynb` | 手搭 dLLM, 在回文上看**并行迭代解码** + 并行度vs质量旋钮 |
| `notebooks/N2-dllm-vs-ar.ipynb` | dLLM vs AR: 双向 **infilling** dLLM 准、AR 瞎猜 (杀手锏) |

## 工具 (`src/`)
- `diffusion_lm.py` — masked diffusion LM (双向 transformer + 置信度并行解码) + AR 对照模型 + infilling 评测 (torch CPU)

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 全部通过 ✅
```
Python 3.13 / torch (tiny CPU) / numpy / matplotlib。玩具回文语言离线确定性。

> **为什么用回文当玩具语言?** 回文 (位置 i = 位置 L-1-i) 让 dLLM 的双向优势**可证伪**: 挖掉中间靠左的位 (镜像在右侧), AR 因果只看左文→瞎猜, dLLM 看双侧→准。机制和真 dLLM 一致, 只是"语言"极简。

## 完成本专题后你应该能
- [ ] 解释 masked diffusion LM 的前向 (遮盖) 与生成 (并行迭代解码)
- [ ] 对比 AR vs dLLM: 因果单向 vs 双向并行, 各自优劣
- [ ] 解释并行度 vs 质量旋钮 (呼应扩散步数 M13.2)
- [ ] 说清 dLLM 的双向 infilling 为何是结构性优势
- [ ] 用 M9.3 批判 dLLM 的现状与 open 问题

---
## 在 Module 13 中的位置
```
  13.1→13.2→13.3→13.4 视频 → 13.5 世界模型 → 13.6 dLLM ◄你在这 → 13.7 capstone
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module13-diffusion-world-models-design.md`
