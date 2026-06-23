# 10.4 visual-tokenization-generation — 让 VLM 不只读图, 还能画图

> **Module 10「多模态/VLM 基础」· 第 4 专题 (从理解到生成)**
> M10.3 的 VLM 会「读」图。本专题让它「画」图: 把图变成离散 token (VQ), 像生成文本一样自回归生成视觉 token, 通往 any-to-any。

---

## 这个专题要解决的真问题

LLM 只会生成离散 token, 图像是连续的 —— 怎么让 LLM 画图?

```
   图 → VQ 离散化 → 视觉 token (像视觉的词) → 自回归生成 → 解码成图
```

- **L1 离散化**: VQ-VAE/VQGAN 学一个视觉码本, 把图量化成离散 token。
- **L2 生成**: 生成图 = 自回归生成视觉 token (复用全部文本生成机器)。
- **L3 理解+生成一体**: 一个模型既读又画 (Chameleon)。
- **L4 any-to-any**: 推到极致, 任意模态互转; 瓶颈在 tokenizer。

> 一条主线: **离散化让所有模态说同一种话 (token)**, 一个 transformer 通吃 —— 多模态统一的钥匙。

## 学习路径 (4 讲)
| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-discrete-visual-tokens.md` | VQ 码本: 图变离散 token, 码本大小权衡 |
| L2 | `lectures/L2-llm-generates-images.md` | 生成图 = 自回归生成视觉 token (同文本机制) |
| L3 | `lectures/L3-unified-understanding-generation.md` | 一个模型既读又画 (Chameleon/Transfusion) |
| L4 | `lectures/L4-any-to-any-tradeoffs.md` | any-to-any 权衡 + 跨模态迁移 (PhD gap) |

## 动手 (2 个 notebook)
| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-vq-reconstruction.ipynb` | 用 `src/vq_tokenizer.py` 把合成图量化成离散 token 再重建, 画「码本大小→重建质量」曲线 |
| `notebooks/N2-autoregressive-generation.ipynb` | 训 tiny transformer 预测下一视觉 token, 采样生成新 token 串、解码成图 — 让「LLM 画图」 |

## 工具 (`src/`)
- `vq_tokenizer.py` — VQ 码本 (numpy k-means) + 量化/反量化/重建, 离线确定性

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 全部通过 ✅
```
Python 3.13 / numpy (VQ) / torch (N2 自回归, tiny CPU) / matplotlib。合成图离线确定性。

## 完成本专题后你应该能
- [ ] 解释 VQ 怎么把图变离散 token + 码本大小权衡
- [ ] 说清「生成图 = 自回归生成视觉 token」与文本生成同机制
- [ ] 区分自回归 vs 扩散两条生成路, 及混合 (Transfusion)
- [ ] 说清理解+生成一体 (Chameleon) 与 any-to-any 的权衡
- [ ] 给 any-to-any 扫出研究 gap

---
## 在 Module 10 中的位置
```
  10.1→10.2→10.3 (读图) → 10.4 (画图/生成) ◄你在这 → 10.5 视频音频 → 10.6 评测 → 10.7 capstone
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module10-multimodal-vlm-design.md`
