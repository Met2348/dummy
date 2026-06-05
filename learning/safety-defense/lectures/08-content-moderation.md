# L08 · Content moderation 传统方法

## OpenAI Moderation API

```python
import openai
resp = openai.moderations.create(input="Some text")
# {
#   "categories": {
#     "hate": False, "hate/threatening": False,
#     "self-harm": False, "sexual": False,
#     "sexual/minors": False, "violence": False,
#     ...
#   },
#   "category_scores": { ... },
#   "flagged": False
# }
```

- **免费、低延迟**
- 11 类（最新 omni-moderation 13 类）
- 准确率高但只覆盖 input

## Anthropic 安全设计

无独立 moderation API，所有 safety 在 LLM 内嵌：
- Claude 自带 Constitutional AI
- 不需要外部 classifier 兜底

## Google Perspective API

- 7 类毒性（toxic / severe_toxic / insult / profanity / threat / sexually_explicit / flirtation）
- 用于 YouTube / Stack Overflow 评论审核
- 多语言（10+）

## 与 Llama Guard 区别

| 维度 | Moderation API | Llama Guard 3 |
|------|---------------|----------------|
| 用途 | 文本/UGC 审核 | LLM input/output 审核 |
| 类别 | 11-13 一般 toxicity | 14 LLM-specific |
| 延迟 | 50-200ms | 1-2s (8B) |
| 成本 | 免费 / 按 token | 自部署算力 |
| 自定义 | 不可 | 可微调 |

## 选择策略

```
question: 何用？
├── chat 内容审核 → OpenAI Moderation
├── UGC (评论 / 论坛) → Perspective API
├── LLM 安全护栏 → Llama Guard 3 / ShieldGemma
├── 业务规则 → NeMo Guardrails
└── 顶级 (Claude 等级) → Constitutional Classifiers
```

## 经典数据集

| Dataset | 用途 |
|---------|------|
| Civil Comments | toxicity 训练 |
| Jigsaw Toxic Comments | Kaggle 比赛起家 |
| HateXplain | 仇恨 + 解释 |
| RealToxicityPrompts | LLM 输出 toxicity 评估 |
| ToxiGen | 隐式 + 显式仇恨 |

## 一句话

> 传统 moderation = 文本审核 11 类，LLM safety = LLM-specific 14 类，定位不同。
