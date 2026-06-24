# 13.3 dit-latent-diffusion — 扩散的骨架: DiT + latent + CFG

> **Module 13「扩散/生成式媒体/世界模型」· 第 3 专题 (骨架与控制)**
> M13.1/13.2 讲扩散的机制和速度。本专题讲它的骨架 (DiT, 你的本行 transformer) + 省算力 (latent diffusion) + 控制 (CFG)。这些拼起来 = Stable Diffusion。

---

## 这个专题要解决的真问题

- **去噪网络用什么?** → **DiT (Diffusion Transformer)**: 把 U-Net 换成 transformer (你的本行, scaling 好)。
- **怎么省算力?** → **latent diffusion**: 在 VAE 压缩的 latent 空间扩散 (接 M10.4 VQ/VAE)。
- **怎么控制条件?** → **CFG (classifier-free guidance)**: guidance scale 旋钮控制贴合度。
- **文本怎么接?** → 文本编码 (M10.1) + cross-attn (M10.2) + CFG = 文生图。

> **文生图 = 你已学部件的组合**: 文本塔 + cross-attn + DiT + CFG + latent = Stable Diffusion。你懂了每一块。

## 学习路径 (4 讲)
| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-transformer-for-diffusion.md` | DiT: transformer 当去噪网络 (你的本行, scaling 好) |
| L2 | `lectures/L2-latent-diffusion.md` | 在 VAE 压缩空间扩散省算力 (接 M10.4) |
| L3 | `lectures/L3-classifier-free-guidance.md` | CFG: guidance scale 控制条件贴合 (公式逐项) |
| L4 | `lectures/L4-conditional-generation.md` | 文本接进扩散 = 文生图 (拼起所有部件) |

## 动手 (2 个 notebook)
| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-dit-conditional.ipynb` | 用 `src/dit.py` 在 4 类高斯团上训 DiT, 指定类别生成对应的团 |
| `notebooks/N2-cfg-ablation.ipynb` | CFG 强度消融: guidance 0→4, 看生成从「随便落」收敛到「精确落在指定类」(接 9.4) |

## 工具 (`src/`)
- `dit.py` — DiT 式条件去噪器 (transformer + 时间/类别条件) + CFG 采样 + 类别准确率 (torch CPU)

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 全部通过 ✅
```
Python 3.13 / torch (tiny CPU) / numpy / matplotlib。4 类高斯团离线确定性。

## 完成本专题后你应该能
- [ ] 解释 DiT (transformer 去噪) 为什么胜 U-Net (scaling/全局/多模态条件)
- [ ] 说清 latent diffusion 怎么省算力 (接 M10.4 压缩)
- [ ] 推导 CFG 公式, 解释 guidance scale 的保真 vs 多样权衡
- [ ] 画出文生图完整流水线, 标出每块来自哪个专题
- [ ] 在 4 类数据上做条件生成 + CFG 消融

---
## 在 Module 13 中的位置
```
  13.1 地基 → 13.2 加速 → 13.3 骨架(DiT/latent/CFG) ◄你在这 → 13.4 视频 → 13.5 世界模型 → 13.6 dLLM → 13.7 capstone
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module13-diffusion-world-models-design.md`
