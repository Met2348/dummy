# L12 · 真实世界案例

> 12 slides | 35 min ⭐⭐⭐⭐

## Slide 1 · 5 个标杆模型

```
GPT-4 (闭源, 推测 1.8T MoE)
Claude 3.5 (闭源)
Llama-3 405B (开源 dense)
DeepSeek-V3 671B (开源 MoE)
Qwen-2.5 72B (开源)
```

## Slide 2 · 开源最强 (2025)

```
DeepSeek-V3 (671B / 37B active):
  MMLU 88, GSM8K 89, HumanEval 82
Llama-3-405B:
  MMLU 87, GSM8K 96, HumanEval 89
Qwen-2.5-72B:
  MMLU 84, GSM8K 91, HumanEval 86
```

## Slide 3 · 推理 SOTA (2025)

```
DeepSeek-R1: 推理 AIME 80
Claude 3.7 Extended Thinking: MATH 95+
Gemini 2.5: thinking_budget API
Qwen-Math: 数学专用 80+
```

## Slide 4 · 小模型 (< 10B)

```
Phi-3-mini 3.8B: MMLU 69 (惊人)
Qwen-2.5-7B: MMLU 75
Llama-3-8B: MMLU 66
DeepSeek-Coder-V2-Lite-16B: HumanEval 81
```

## Slide 5 · 端侧 / 1B 级

```
Phi-3-mini 1.3B: MMLU 60+
Qwen-2.5-1.5B: MMLU 60
Llama-3.2-1B: MMLU 49
本 capstone 270M: 期望 MMLU 25-35
```

差距 25-35pp, 主要因 1) token 量 2) post-training.

## Slide 6 · 中文模型

```
Qwen-2.5 系列: 中文 SOTA
DeepSeek-V3: 中文 + 英文双优
ChatGLM4-9B: 中文垂域
Yi-1.5-34B: 通用中文
Baichuan-2-13B: 通用
```

## Slide 7 · 多模态

```
GPT-4o: 主流
Claude 3.5 Sonnet: vision
Gemini 1.5 Pro: 1M ctx
Qwen2-VL: 开源最强
Phi-3.5-vision: 端侧
```

## Slide 8 · 推理模型 (2025)

```
o1-preview (2024.09): 推理 RL 范式起点
o3 (2024.12): SOTA 推理 + 84% on ARC
R1 (2025.01): 开源对标 o1
Claude 3.7 thinking (2025): 商业 thinking model
Gemini 2.5 thinking_budget: 商业可控 thinking
```

## Slide 9 · 训练成本对比

```
GPT-4 (推测): $63M
Llama-3-405B: $14M
DeepSeek-V3: $5.6M ⭐ 工程奇迹
Phi-3 (合成数据): $1M 级
本 capstone: $0.5 GPU 单卡 1 day
```

## Slide 10 · 推理成本对比 (per 1M token)

```
GPT-4o: $5 input / $15 output
Claude 3.5: $3 / $15
Llama-3-70B (HF): $0.9
Qwen-2.5-72B (DeepInfra): $0.5
DeepSeek-V3 (官方): $0.27 / $1.1
```

DeepSeek-V3 推理便宜的主因: MoE (37B active).

## Slide 11 · 选型决策树 (个人)

```
学习: 本 capstone (Phi-tiny 270M)
试: Qwen-2.5-7B local (vLLM)
生产: DeepSeek-V3 API / Llama-3-70B local
专用: 微调小模型 (1B 级 + LoRA)
推理: Claude 3.7 thinking / R1
```

## Slide 12 · 总结

```
2026 SOTA 开源 = DeepSeek-V3 671B + R1 + Qwen-2.5
SOTA 闭源 = GPT-4 + Claude 3.7 + Gemini 2.5
端侧 = Phi-3.5 / Qwen-2.5-3B
本 capstone = 学习用 270M
```

## 参考
- 各模型 tech report
- HF leaderboards
