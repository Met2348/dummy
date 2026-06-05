# L01 · 防御 4 层结构

## 总图

```
USER ─┐
      │
      ▼
[L1] Input Classifier  (Llama Guard 3 / Constitutional)
      │
      ▼
[L2] System prompt + RLHF (训练时的安全)
      │
      ▼
   LLM 推理
      │
      ▼
[L3] Output Classifier  (Llama Guard 3 / Llama Guard Output)
      │
      ▼
[L4] Rule-based guardrails (NeMo / 业务规则)
      │
      ▼
USER ←┘
```

## 4 层职责

| 层 | 谁拦 | 优 | 缺 |
|----|------|-----|-----|
| L1 input | 直接 harmful query | 早拦省钱 | 易绕（隐式表达）|
| L2 训练 | 大部分 jailbreak | 鲁棒 | 训练慢 + 不可在线改 |
| L3 output | 漏网的有害输出 | 兜底 | 算力 ×2 |
| L4 rule | 业务特定 | 可定制 | 维护重 |

## 防御层数 vs ASR

```
no defense                   → ASR 80%
+ L1 input classifier        → ASR 30%
+ L2 strong RLHF             → ASR 10%
+ L3 output classifier       → ASR 3%
+ L4 rule-based              → ASR 1%
```

→ **每加一层 ASR 减 2-5×**。

## 业界实践

| 公司 | 主防御 |
|------|--------|
| Anthropic | Constitutional AI (L2) + Constitutional Classifiers (L1+L3) |
| OpenAI | RLHF (L2) + Moderation API (L1+L3) |
| Meta | RLHF (L2) + Llama Guard 3 (L1+L3) |
| Google | Gemini Safety (L2) + Shield Gemma (L1+L3) |
| NVIDIA | NeMo Guardrails (L4) for enterprise |

## 本 Topic 覆盖

L02-L05: 4 大开源 / 商业 classifier
L06: Constitutional Classifiers (Anthropic 2025) ⭐
L07: Prompt injection 防御
L08-L09: PII + 内容审核
L10: Monitoring + 事件响应
L11: 安全 bench
L12: Capstone 4-layer pipeline

## 一句话

> 安全防御 = 4 层防御纵深，单层都易被绕，combo 才稳。
