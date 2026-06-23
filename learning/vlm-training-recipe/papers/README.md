# papers/ — vlm-training-recipe 参考源

## VLM 训练配方
- **LLaVA** — "Visual Instruction Tuning" (Liu et al. 2023) + **LLaVA-1.5** (改进配方). 两阶段配方 + GPT-4 合成指令数据 (L1/L3)。
- **Qwen-VL / Qwen2-VL** — 高分辨率 + 强配方。
- **InternVL** — 更大视觉塔 + 对齐策略。
- **Flamingo** (训练侧: 交错文档预训练, L2)。

## 数据 / 合成
- LLaVA 的 GPT-4 视觉指令数据合成 (合成数据在 VLM 的经典应用, 接 data-curation)。
- LAION / CC 等图文对数据集 (阶段 1 对齐)。

## 训练陷阱 / 平衡
- 关于灾难遗忘 / 模态平衡的讨论散见各 VLM 技术报告 (L4); 用 9.3 批判式读它们的「我们如何避免遗忘」。
- LoRA (你 M1 PEFT) 在 VLM 微调里抗遗忘的应用。

> 本专题知识在可跑的 `mini_vlm.py` (端到端训练 + 冻结开关) 里。
> 真练习: 用 HuggingFace 复现 LLaVA-1.5 的阶段 1 (只训投影器), 体会「只训一座小桥」多便宜。
