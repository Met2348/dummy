# papers/ — vlm-eval-hallucination 参考源

## VLM benchmark
- **MMMU** — "A Massive Multi-discipline Multimodal Understanding Benchmark" (2023). 大学级推理 (L2)。
- **MME** — VLM 多维评测 (14 子任务, L2)。
- **POPE** — "Evaluating Object Hallucination in Large Vision-Language Models" (Li et al. 2023). yes-bias 幻觉探测 (L2/L3, 本专题核心)。
- MathVista / OCRBench / RefCOCO — 数学/OCR/定位维度 (L2)。

## 视觉幻觉机理与缓解
- CHAIR — 描述里「图中没有的物体」占比指标 (L3)。
- 对比解码 (contrastive decoding) 抗幻觉 (L3)。
- **DPO/RLHF 抗幻觉** — 用诚实当偏好信号 (接你的 dpo-family, L3 gap)。

## 诚实评测 (接 Module 9)
- 9.3 攻击清单 / 9.4 公平对照 / 9.6 诚实图 在 VLM 评测的应用 (L4)。
- 你的 M6 评测安全专题 (eval-foundations/judge-arena) 的多模态延续。

> 本专题知识在可跑的 `vlm_eval.py` (POPE + 模拟 VLM) 里。
> 真练习: 拿 HuggingFace 一个真 VLM (如 LLaVA) 跑一遍 POPE, 看它的真实 yes-rate。
