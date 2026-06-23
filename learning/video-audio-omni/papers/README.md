# papers/ — video-audio-omni 参考源

## 视频
- **ViViT** — "A Video Vision Transformer" (Arnab et al. 2021). 时空 patch + 分解注意力 (L1)。
- **VideoMAE** — 视频自监督。
- **Sora** (技术报告) — 时空 patch latent + 扩散 (接 M13 视频生成)。

## 音频 / 语音
- **Whisper** — "Robust Speech Recognition via Large-Scale Weak Supervision" (Radford et al. 2022). mel 谱输入 (L2)。
- **HuBERT / wav2vec 2.0** — 自监督离散语音单元 (L2)。
- **AudioLM / MusicGen** — 音频 token 自回归生成。

## omni / 实时
- **Gemini / GPT-4o** (技术报告/博客) — omni 模型 + 实时语音 (L3/L4)。
- **Qwen-Omni / AnyGPT** — 开源 any-to-any (L3)。
- 实时约束接你的 speculative-decoding / quantization (M5) + harness-engineering 专题。

> 你的 EE 背景 (信号处理/FFT/滤波) 在音频 (L2) 是直接优势。
> 本专题知识在可跑的 `temporal_tokens.py` / `audio_features.py` (纯 numpy) 里。
