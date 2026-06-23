# papers/ — visual-tokenization-generation 参考源

## 视觉 token 化 (VQ)
- **VQ-VAE** — "Neural Discrete Representation Learning" (van den Oord et al. 2017). 离散码本 (L1)。
- **VQGAN** — "Taming Transformers for High-Resolution Image Synthesis" (Esser et al. 2021). VQ + GAN/感知损失 (L1)。

## 自回归图像生成
- **DALL-E 1** — 离散 token 自回归生成图 (L2)。
- **Parti** — 自回归文生图的 scaling。

## 理解 + 生成一体 / any-to-any
- **Chameleon** — "Mixed-Modal Early-Fusion Foundation Models" (Meta 2024). 纯离散 token 统一 (L3)。
- **Transfusion** — 自回归 + 扩散混合 (Meta 2024). 文本自回归 + 图像扩散 (L3, 接 M13)。
- **Emu / SEED / AnyGPT** — any-to-any 多模态探索 (L4)。

## 与 M13 的桥
- 图像生成的另一条路 (扩散) 是 M13 整个模块; 本专题讲自回归 token 路, 两条都要懂。

> 用 9.3 批判式读: any-to-any 的「跨模态迁移」假设是否被验证? 统一真比专精好吗? (L4 的 gap)。
> 本专题知识在可跑的 `vq_tokenizer.py` (VQ 重建 + 离散 token) 里。
