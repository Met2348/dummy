# papers/ — vision-encoders 参考源

## 视觉 Transformer
- **ViT** — "An Image is Worth 16x16 Words" (Dosovitskiy et al. 2020). 图像即 patch 序列的开山之作 (L1)。
- DeiT / Swin Transformer — ViT 的高效/层次化变体。

## 图文对比学习
- **CLIP** — "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al. 2021). InfoNCE 图文对比 (L2)。
- **SigLIP** — "Sigmoid Loss for Language Image Pre-Training" (Zhai et al. 2023). 成对 sigmoid 损失 (L3)。

## 自监督视觉
- **DINOv2** — "DINOv2: Learning Robust Visual Features without Supervision" (Oquab et al. 2023). 自蒸馏 (L3)。
- MAE — "Masked Autoencoders Are Scalable Vision Learners" (另一条自监督路, 掩码重建)。

## VLM 里的视觉塔用法
- LLaVA (用冻结 CLIP 视觉塔 + MLP 连接器) — 接 10.2/10.3。
- Flamingo (perceiver resampler 压缩视觉 token) — 接 10.2。

> 用 9.3 批判式读这些: 对比训练的目标决定了表示的盲区 (L4 的「最后层不总最好」就是一例)。
> 本专题知识在可跑的 `tiny_vit.py`/`contrastive.py` 里 —— 真练习是拿真 CLIP/SigLIP (HuggingFace) 跑一遍 zero-shot 分类。
