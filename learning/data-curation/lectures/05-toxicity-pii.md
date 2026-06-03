# L05 · 毒性 + PII — Detoxify 与 Presidio

> 20 slides | 60 min | Data Curation 第 5 讲 ⭐⭐⭐⭐

---

## 学习目标

1. 理解 toxicity 在 LLM 训练数据中的影响
2. 用 Detoxify multi-label classifier 过滤毒性
3. 用 Presidio analyzer 检测 PII
4. 用 anonymizer 替换 PII（vs 删除）
5. 知道 GDPR / CCPA 对训练数据的硬性要求

---

## Slide 1 · 为什么过滤 toxicity？

LLM 在 web 上训练会学到：
- 仇恨言论
- 自残 / 自杀引导
- 暴力描述
- 性别 / 种族偏见

直接后果：
- API 部署时被 jailbreak
- 用户体验灾难
- 法律风险

---

## Slide 2 · 三层防御

```
1. 数据层    → 过滤 / 替换 toxicity / PII
2. 训练层    → RLHF / Constitutional AI
3. 推理层    → guardrails / safety filter
```

数据层是"成本最低、收益最持久"的防御。本课主讲数据层。

---

## Slide 3 · Detoxify 简介

unitary.ai 开源：

- BERT-based multi-label classifier
- 标签：toxicity, severe_toxicity, obscene, threat, insult, identity_attack, sexual_explicit
- 训练数据：Jigsaw Toxic Comment Challenge

```python
from detoxify import Detoxify
scores = Detoxify("original").predict("I hate you idiot")
# {"toxicity": 0.98, "insult": 0.95, ...}
```

---

## Slide 4 · Detoxify 模型选择

| 模型 | 数据 | 大小 | 推荐 |
|------|------|------|------|
| `original` | Jigsaw 1.0 | 12L BERT | 通用 |
| `unbiased` | + bias 补 | 12L | 减少误判 |
| `multilingual` | 7 语言 | XLM-R | 非英语 |

教学用 `original`，生产用 `unbiased` 防误删少数群体讨论。

---

## Slide 5 · 阈值的选择

```
threshold = 0.5  → 中度过滤 (FineWeb 风格)
threshold = 0.8  → 宽松，少误删
threshold = 0.3  → 严格，多误删
```

实务：toxicity ≥ 0.5 或 severe_toxicity ≥ 0.3 任一触发就丢。

---

## Slide 6 · "删除"vs "保留 mask"

| 策略 | 优 | 劣 |
|------|-----|----|
| 直接删 | 简单 / 隐患小 | 模型不学"何为 toxic" |
| 保留 + 训练 + RLHF 防止生成 | 模型理解 | 复杂、需要 RLHF |
| mask 替换 | 折中 | 信号弱 |

教学版用直接删（最稳）。

---

## Slide 7 · 误判 / 偏见

Detoxify 早期模型会把：
- 关于少数群体的 **讨论** 误判为 attack
- 非英语翻译误判为 toxic
- 学术分析判为 explicit

`unbiased` 模型缓解 30%；仍需 ablation。

---

## Slide 8 · PII (Personally Identifiable Information)

| 类别 | 例 |
|------|-----|
| 直接 | 姓名、电话、邮箱 |
| 间接 | 地址、生日 |
| 敏感 | 身份证、信用卡、医疗号 |
| 在线 | IP、设备 ID |

GDPR / CCPA / China PIPL 都要求训练数据合规。

---

## Slide 9 · Presidio 简介

Microsoft 开源：

```
presidio-analyzer    → 检测（NER + regex + spaCy）
presidio-anonymizer  → 替换 / 删除 / hash
```

支持 30+ 种 PII：EMAIL, PHONE, SSN, CREDIT_CARD, IP, PERSON, LOCATION...

---

## Slide 10 · Presidio 分析

```python
from presidio_analyzer import AnalyzerEngine
analyzer = AnalyzerEngine()
results = analyzer.analyze(
    text="My email is john@example.com",
    entities=["EMAIL_ADDRESS", "PHONE_NUMBER"],
    language="en",
)
# [RecognizerResult(entity=EMAIL_ADDRESS, start=12, end=30, score=1.0)]
```

返回 entity 范围，可定制识别器。

---

## Slide 11 · Presidio 替换

```python
from presidio_anonymizer import AnonymizerEngine
anonymizer = AnonymizerEngine()
result = anonymizer.anonymize(
    text="My email is john@example.com",
    analyzer_results=results,
)
# "My email is <EMAIL_ADDRESS>"
```

可改 placeholder（`<EMAIL>`）或加 hash / encrypt。

---

## Slide 12 · 替换策略

| 替换形 | 例 |
|--------|-----|
| placeholder | `<EMAIL>` |
| 假数据 | `john@example.com` |
| hash | `<EMAIL_a1b2c3>` |
| 删除 | (整段删除) |

placeholder 训练效果最好（模型仍学"这里是 PII"）。

---

## Slide 13 · 双语 PII

Presidio 中文支持有限，需自训。

```python
# 简单 regex 兜底
import re
PHONE_CN = re.compile(r"1[3-9]\d{9}")
ID_CN = re.compile(r"\d{17}[\dXx]")
```

中文 / 日文 / 韩文 PII 需另外补 regex + spaCy 模型。

---

## Slide 14 · 误判与 dataloss

PII 检测会误删：
- 历史人物姓名（"Albert Einstein"）
- 公司名误判 ORGANIZATION
- 简单数字误判 SSN

→ 配 confidence 阈值 0.6+ 或 entity allowlist。

---

## Slide 15 · 实务管线

```
text
 ├─ Detoxify(score) ──→ keep / drop
 │
 ├─ Presidio analyzer ─→ [entities]
 │      ↓
 │   anonymizer ─→ text_replaced
```

输出 jsonl 含 `text_clean` + 原始 metadata。

---

## Slide 16 · 性能

| | 速度 | GPU 加速 |
|---|------|---------|
| Detoxify | ~200 doc/s | yes (BERT) |
| Presidio | ~500 doc/s | partial (spaCy) |

1B doc 量级 → 多 GPU + Ray / Dask。

---

## Slide 17 · 法律视角

```
GDPR  ──→ "right to be forgotten" → 训练数据需可追溯删除
CCPA  ──→ 加州类似要求
PIPL  ──→ 中国，更严
```

实务：保留 doc_id ↔ url 映射，紧急情况删 doc 后增量再训。

---

## Slide 18 · 关于"已发布"数据

公开 web 的 PII 仍可能被诉。Llama-2 / Mistral / Phi-3 都被起诉过。

防御：
1. PII 过滤
2. Constitutional AI / RLHF 训不生成 PII
3. Inference-time PII filter

---

## Slide 19 · 评估指标

```
PII 替换率 = #replaced / #true_PII   ≥ 90%
误判率    = #false_positive / #total  ≤ 5%
```

FineWeb 自报 ~92% / 4%。

---

## Slide 20 · 课后思考

1. toxicity 与 quality 是同一维度吗？
2. PII 检测在多语言下为什么难？
3. 如果删除 toxic 数据，模型还能识别 hate speech 吗？
4. 是否应该保留 toxic 数据但 RLHF 训练时强压制？

---

## 参考

- Detoxify: https://github.com/unitaryai/detoxify
- Presidio: https://microsoft.github.io/presidio/
- GDPR Art. 17 (Right to erasure)
- Jigsaw Toxic Comment Challenge
