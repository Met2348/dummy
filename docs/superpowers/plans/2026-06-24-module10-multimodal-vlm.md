# Module 10 多模态/VLM Implementation Plan

**Goal**: 新建 Module 10「多模态/VLM 基础」(7 专题), 从纯文本 LLM 跨到视觉-语言基础模型。每专题统一外壳 `README + lectures + notebooks + templates + src + environment`, notebook 小尺度 CPU/单卡可跑、nbconvert 0 报错。

**Design**: `docs/superpowers/specs/2026-06-24-module10-multimodal-vlm-design.md`

**Tech Stack**: Python 3.13 / torch (tiny 模型 CPU 可跑) / numpy / matplotlib / nbformat / pandas. 复用 Module 9 的可复现纪律 (确定性 seed, 离线样例/合成数据)。

**构建顺序 (模块内)**: 10.1 encoder → 10.2 fusion → 10.3 train → 10.4 generation → 10.5 video/audio → 10.6 eval → 10.7 capstone (后者依赖前者的 src)。

**在 M10-M13 全局的位置**: **第 1 个建** (是 M11/M13 的地基)。

---

## Phases (每专题一个 phase, 内部 6 步)
每专题: ① 目录骨架 + environment(requirements+verify_env) ② README ③ lectures (4, capstone 2) ④ src 工具 (含离线/合成数据) ⑤ templates ⑥ notebooks (nbformat 生成 → nbconvert 执行验证) → commit。

- **P1 `vision-encoders`**: tiny_vit.py / contrastive.py; N1 ViT patch embed / N2 mini-CLIP
- **P2 `vl-fusion-architectures`**: connectors.py (3 种连接器); N1 连接器对比 / N2 LLaVA 式投影
- **P3 `vlm-training-recipe`**: mini_vlm.py; N1 端到端训 mini-VLM / N2 冻结策略消融 (复用 9.5 tracker)
- **P4 `visual-tokenization-generation`**: vq_tokenizer.py; N1 VQ 重建 / N2 视觉 token 自回归生成
- **P5 `video-audio-omni`**: temporal_tokens.py / audio_features.py; N1 时空 token / N2 mel→token
- **P6 `vlm-eval-hallucination`**: vlm_eval.py; N1 POPE 式幻觉探测 / N2 评测出图 (复用 9.6)
- **P7 `multimodal-graduation`**: Capstone; N1 端到端 mini-VLM 问答 / N2 多模态 gap→idea 卡 (复用 9.3/9.4)

## 成功标准
- [ ] 7 专题完整, verify_env 全过, 14 notebook nbconvert 0 报错。
- [ ] 至少一个 notebook 端到端训出可问答 mini-VLM。
- [ ] 课件公式逐项 (InfoNCE/cross-attn/VQ)。
- [ ] portfolio 体现「文本→多模态」。
