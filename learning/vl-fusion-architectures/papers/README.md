# papers/ — vl-fusion-architectures 参考源

## 投影路线
- **LLaVA** — "Visual Instruction Tuning" (Liu et al. 2023). MLP 投影连接器 (L3)。
- LLaVA-1.5 / LLaVA-NeXT — 投影路线的演进 (更高分辨率/更强)。

## cross-attention 路线
- **Flamingo** — "a Visual Language Model for Few-Shot Learning" (Alayrac et al. 2022). perceiver resampler + 门控 cross-attn (L2)。
- BLIP-2 — Q-Former (另一种压缩 + 桥接, 介于投影和 cross-attn)。

## early-fusion 路线
- **Chameleon** — "Mixed-Modal Early-Fusion Foundation Models" (Meta 2024). 统一 token, 理解+生成一体 (L4)。
- **Fuyu** — patch 直接当 token 的 early-fusion (Adept)。
- Transfusion — 自回归 + 扩散混合 (接 M13)。

> 用 9.3 批判式读: 每条路线优化什么、盲区在哪 (如投影路线的上下文成本、early-fusion 的从头训成本)。
> 本专题知识在可跑的 `connectors.py` 里 —— 真练习是拿 HuggingFace 的 LLaVA 跑一遍, 看它的投影器多简单。
