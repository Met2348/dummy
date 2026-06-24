# papers/ — dit-latent-diffusion 参考源

## DiT
- **DiT** — "Scalable Diffusion Models with Transformers" (Peebles & Xie 2023). transformer 去噪 + scaling (L1)。
- **SD3 / Flux** — 工业级 DiT + flow matching (L1)。

## latent diffusion
- **Stable Diffusion / LDM** — "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al. 2022). VAE latent 扩散 (L2)。

## classifier-free guidance
- **CFG** — "Classifier-Free Diffusion Guidance" (Ho & Salimans 2022). L3。
- ControlNet — 结构条件控制 (L4)。

## 与你专题的桥
- 文本编码 ← M10.1 CLIP 文本塔; cross-attn 注入 ← M10.2; latent 压缩 ← M10.4 VQ/VAE。
> 本专题知识在可跑的 `dit.py` (DiT + CFG) 里。真练习: 跑一个真 Stable Diffusion, 调 CFG scale 看效果。
