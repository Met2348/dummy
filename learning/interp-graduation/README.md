# 12.7 interp-graduation — Module 12 Capstone

> **Module 12「机制可解释性」· 第 7 专题 (毕业设计)** 🎓
> 把 M12 全套工具串成**一次完整逆向工程** (探针→patching→SAE→电路), 对一个模型产出连贯机制故事; 并用 M9 找 gap 框架产出 **interp × reasoning** 研究 idea 卡 (你最可能转 PhD 题)。

---

## 这个专题要做的两件事
1. **完整 interp 流程**: import M12 全套 src, 对玩具 transformer 跑 探针→patching→SAE, 产出连贯、因果、有证据的机制故事。
2. **找研究 gap**: 5 个 interp gap (2 个 ★ interp×reasoning), 产 idea 卡, 挑最匹配你的细化成 PhD 种子。

## 学习路径 (2 讲, capstone)
| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-full-interp-workflow.md` | 把 12.1-12.6 串成一次完整逆向工程 (证据链) |
| L2 | `lectures/L2-frontiers-and-your-path.md` | interp 前沿 + 你的路径 (interp×reasoning PhD) |

## 动手 (2 个 notebook)
| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-full-interp.ipynb` | 装配 M12 全套 + 完整流程 (探针→patching→SAE 证据链) |
| `notebooks/N2-research-idea-cards.ipynb` | 5 张 interp idea 卡, 高亮 ★ interp×reasoning |

## 工具 (`src/`)
- `interp_capstone.py` — 跨专题装配 (M12 全套) + run_full_interp (完整流程) + gap 雷达 + idea 卡。

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 全部通过 ✅ (含 M12 全套装配 + 完整流程)
```

## 完成本专题后你应该能
- [ ] 对一个模型跑完整逆向工程 (探针→patching→SAE), 产出连贯机制故事
- [ ] 说清 interp 研究前沿 (SAE/自动circuit/interp×安全/interp×reasoning)
- [ ] 在 interp 上找 gap, 写可执行 idea 卡 (可证伪假设 + 最小实验)
- [ ] 说清为什么 interp×reasoning 是你 (NLP+EE+reasoning) 的 PhD 甜点

---
## 在 Module 12 中的位置
```
  12.1→...→12.6 → 12.7 capstone ◄你在这 (Module 12 毕业 🎓)
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module12-interpretability-design.md`
> 你的 PhD 种子: interp × reasoning (CoT 忠实性的机制级验证)。
