# L11 · 选型决策树

## 5 ckpt 选型

```
question: 我的场景是？

├── 知识 QA (轻量)
│       └── lora (124M) 或 phi_tiny (270M)
│
├── 推理 / 数学 / 代码
│       ├── 需要思考链 → r1_tiny ($$, latency 高)
│       └── 简洁推理 → phi_tiny
│
├── 安全敏感 (客服 / 政务 / 教育)
│       └── dpo + 4 层防御 (Topic 6)
│
├── 多模态
│       └── 需 VLM ckpt (Module 4 multimodal-agent)
│
├── 长 ctx (RAG / 文档)
│       └── 需长 ctx ckpt (Module 3 long-context)
│
└── 端侧 (≤1GB RAM)
        └── phi_tiny 4bit quant (Module 5 quantization-deploy)
```

## bench 选型

```
question: 我想测什么？

├── 知识     → MMLU / MMLU-Pro
├── 推理     → GSM8K / MATH / AIME / GPQA / HLE
├── 代码     → HumanEval / LiveCodeBench
├── Agent    → SWE-Bench / WebArena / OSWorld
├── 对话     → MT-Bench / Arena-Hard / Chatbot Arena
├── 多模态   → MMMU / MathVista
├── 安全     → HarmBench / JailbreakBench
├── 真实流量 → A/B test live
└── 个人 ↓ 综合 → mini-HELM (本系列)
```

## judge 选型

```
question: 我想用什么 judge？

├── 公开发表 / leaderboard → GPT-4 / Claude
├── 大规模 (10k+) cheap   → Prometheus 2
├── 业务对齐               → fine-tuned domain judge
└── 在线监控               → Llama Guard 3 (低延迟)
```

## 防御选型

```
question: 我担心什么风险？

├── jailbreak / harmful content → Constitutional Classifiers / Llama Guard
├── prompt injection           → 输入 parse + sandwich
├── PII leak                   → Presidio + regex
├── 长 ctx 攻击                → many-shot defense
├── 多模态                     → OCR + image moderation
└── 端到端                     → Topic 6 4-layer pipeline
```

## 部署选型

```
question: 我要部署多少 QPS？

├── 1-10  QPS  → llama.cpp / Ollama
├── 100  QPS  → vLLM 单机
├── 10k  QPS  → vLLM + TP + 多副本
└── 100k QPS  → Disaggregated Prefill/Decode (DistServe)
```

## 一句话

> 决策树 = 把 32 专题学到的"选哪个"压成 5 张快查表。
