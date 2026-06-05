# L09 · PII 检测 + 脱敏

## 为什么必要

LLM 训练 / 推理过程可能：
- 在 logs 里存用户 PII
- 在输出里复述（leak）训练数据中的 PII
- 把 PII 发到第三方 API

GDPR / HIPAA / CCPA 都要求 PII 处理。

## 5 大 PII 类

| 类 | 例 |
|----|-----|
| **email** | alice@example.com |
| **phone** | 415-555-1234 |
| **SSN** | 123-45-6789 |
| **credit card** | 4111-1111-1111-1111 |
| **person name** | Alice Smith |
| **address** | 123 Main St. |
| **IP** | 192.168.0.1 |
| **medical record** | DOB 1985-03-12 |

## 工具栈

### 1. Regex (简单 PII)

```python
import re
PHONE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
EMAIL = re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b")
SSN   = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
```

适合：固定格式 PII（email/phone/SSN）。

### 2. NER (model-based)

```python
from transformers import pipeline
ner = pipeline("ner", model="dslim/bert-base-NER")
results = ner("Alice Smith from San Francisco called.")
# [{'entity': 'B-PER', 'word': 'Alice', ...}, ...]
```

适合：人名/地名/机构（无固定格式）。

### 3. Microsoft Presidio

```python
from presidio_analyzer import AnalyzerEngine
analyzer = AnalyzerEngine()
results = analyzer.analyze(text=..., entities=["PHONE_NUMBER", "EMAIL", ...],
                            language='en')
```

生产推荐：covers 26 entities + supports anonymization。

### 4. Domain-specific (medical, legal)

PHI (Personal Health Info) → 用 [Stanza](https://stanfordnlp.github.io/stanza/) + clinical NER。

## 脱敏策略

| 策略 | 例 | 用途 |
|------|-----|------|
| **redact** | `[EMAIL]` | 默认 |
| **mask** | `***.***@***.com` | 保留结构 |
| **fake** | `fake@example.com` | 测试数据 |
| **encrypt** | 加密保留可逆 | 审计 |
| **hash** | sha256 截断 | 去识别 |

## 实测：Presidio + redact

```
原: "Alice (123-45-6789) called from 415-555-1234."
脱: "[NAME] ([SSN]) called from [PHONE]."
```

## 实操（mock）

src/pii_redaction.py 5 类 regex + 10 个 toy 名字：

```python
from pii_redaction import detect_pii, redact

text = "Alice at alice@x.com, SSN 123-45-6789"
print(detect_pii(text))
# {'email': ['alice@x.com'], 'ssn_us': ['123-45-6789'], 'person_name': ['Alice']}

print(redact(text))
# "[NAME] at [EMAIL], SSN [SSN]"
```

## 一句话

> PII 脱敏 = legal 强制，regex 起步 + Presidio 上生产。
