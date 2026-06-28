# 12.4 sparse-autoencoders — SAE 解叠加, 提单义特征

> **Module 12「机制可解释性」· 第 4 专题 (解叠加)**
> 回到 superposition: SAE (稀疏自编码器) 用**过完备+稀疏**字典, 把叠加的多义神经元**解开成单义特征** (Anthropic 的「显微镜」/金门大桥特征)。

---

## 这个专题要解决的真问题
- **为什么需要 SAE?** → superposition 让神经元多义, 沿坐标轴读串台 → 要解叠加。
- **SAE 原理?** → encode(ReLU,过完备) → decode(字典); loss = 重建 + L1 稀疏。
- **特征单义吗?** → 最大激活样本贴标签; 纯度 >> 原始神经元 (解叠加硬证据)。
- **争议?** → 特征真实吗/评估无标准/单义≠因果在用 (批判, 接 M9.3)。

> **小而真**: 玩具 SAE 纯度 ~4× 原始神经元; 真 gpt2 SAE 特征比多义神经元成主题。

## 学习路径 (4 讲)
| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-superposition-to-sae.md` | superposition→为什么 SAE (过完备+稀疏解叠加) |
| L2 | `lectures/L2-sae-mechanism.md` | SAE 结构+损失逐项 (重建+L1稀疏) |
| L3 | `lectures/L3-monosemantic-features.md` | 单义特征 (最大激活样本/金门大桥) |
| L4 | `lectures/L4-sae-evaluation-debates.md` | SAE 评估与争议 (批判, 接 M9.3) |

## 动手 (2 个 notebook)
| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-train-mini-sae.ipynb` | 训 mini-SAE, 看 SAE 特征纯度 >> 原始神经元 (解叠加) |
| `notebooks/N2-feature-monosemanticity.ipynb` | 特征×值热图(单义) + 真 gpt2 SAE 特征最大激活 token |

## 工具 (`src/`)
- `sae.py` — 稀疏自编码器 (过完备+L1) + 单义性度量; 复用 12.1 tiny_transformer, 也用于真 gpt2。

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 全部通过 ✅
```

## 完成本专题后你应该能
- [ ] 解释为什么 superposition 需要 SAE (过完备+稀疏解叠加)
- [ ] 写出 SAE 结构和损失 (重建 + L1 稀疏), 训一个
- [ ] 用最大激活样本判断特征是否单义 (纯度 vs 原始神经元)
- [ ] 用 M9.3 批判 SAE (真实性/评估/因果/完备性)

---
## 在 Module 12 中的位置
```
  12.1→12.2→12.3 → 12.4 SAE ◄你在这 → 12.5 circuits → 12.6 CoT → 12.7 capstone
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module12-interpretability-design.md`
