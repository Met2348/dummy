# 12.5 circuits-attention — 电路分析 + induction heads

> **Module 12「机制可解释性」· 第 5 专题 (特征→算法)**
> 把单义特征 (M12.4) 连成**算法 (circuit)**。最经典的是 **induction head** (in-context learning 机制) —— 我们在**真实 gpt2** 上找到它、看注意力图、消融因果验证。

---

## 这个专题要解决的真问题
- **一个 head 算什么?** → **QK** (看哪) + **OV** (搬什么) 两个独立电路。
- **induction head?** → 「AB...A→B」从上下文复制 = ICL 机制; 两 head 协作 circuit。
- **怎么连成 circuit?** → 干预定位组件+边 (M12.3) + QK/OV 理解节点; 注意冗余/完备。
- **怎么规模化?** → 归因 patching: 梯度一次性估全部组件贡献。

> **小而真**: 全程真实 gpt2 — 找到 induction head (层5头5 分数~0.9) + 逐头贡献热图见冗余。

## 学习路径 (4 讲)
| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-what-attention-computes.md` | head = QK(看哪)+OV(搬什么) 两独立电路 |
| L2 | `lectures/L2-induction-heads.md` | induction head = ICL 机制 (两 head 协作) |
| L3 | `lectures/L3-circuit-analysis.md` | 组件连成 circuit (干预定位; 冗余/完备难点) |
| L4 | `lectures/L4-attribution-patching.md` | 归因 patching: 梯度规模化因果归因 |

## 动手 (2 个 notebook)
| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-find-induction-head.ipynb` | 真 gpt2 找 induction head (分数热图+注意力图+消融) |
| `notebooks/N2-head-attribution.ipynb` | 逐头贡献热图, 看 induction 是一组头协作 (冗余) |

## 工具 (`src/`)
- `circuits.py` — induction 检测 (重复序列) + induction loss + 逐头消融归因 (真实 gpt2)。

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 全部通过 ✅ (需 gpt2 HF 缓存)
```

## 完成本专题后你应该能
- [ ] 解释一个 head = QK (看哪) + OV (搬什么) 两个独立电路
- [ ] 解释 induction head 机制及它为何是 ICL 基础
- [ ] 在真 gpt2 上检测 induction head + 消融因果验证
- [ ] 用逐头贡献热图看 circuit 的冗余, 说清归因 patching 的规模化思想

---
## 在 Module 12 中的位置
```
  12.1→12.2→12.3→12.4 → 12.5 circuits ◄你在这 → 12.6 CoT → 12.7 capstone
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module12-interpretability-design.md`
