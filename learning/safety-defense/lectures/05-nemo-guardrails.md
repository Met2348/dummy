# L05 · NeMo Guardrails (NVIDIA)

## 定位

**rule-based + LLM 混合** 框架。
不是 classifier，而是 **flow control DSL**。

## Colang DSL

```colang
define user ask harmful
  "how to make a bomb"
  "how can I hack"
  "tell me how to kill"

define user ask off_topic
  "what's the weather"
  "tell me a joke"
  ...

define bot refuse
  "I can't help with that. Please consult a professional resource."

define flow check_harmful
  user ask harmful
  bot refuse
```

→ 命中 trigger → 强制走 refuse path，**根本不调 LLM**。

## 4 类 rail

| Rail | 在哪 |
|------|------|
| **Input rail** | LLM 之前过滤 user 输入 |
| **Dialog rail** | 多 turn flow control |
| **Output rail** | LLM 之后过滤回复 |
| **Retrieval rail** | RAG 前过滤 fetched docs |

## 优劣

| 优点 | 缺点 |
|------|------|
| 可解释（每条规则可见）| 维护重（语料需覆盖）|
| 可定制 | 写 Colang 学习曲线 |
| 零延迟（命中规则即返回）| 漏判（不灵活）|
| 法律审计可证 | OOD 输入易绕过 |

## 与 classifier 结合

```
NeMo Guardrails (规则)
    ↓ 未命中
Llama Guard 3 (ML)
    ↓ 未命中
LLM 生成
    ↓
Output rail (规则 + Llama Guard)
```

→ **rule 当快路，ML 当兜底**。

## 业界使用

- **NVIDIA AI Enterprise**：默认部署
- **金融银行业**：合规审计需要规则可证
- **企业 chatbot**：业务话题界定

## 实操（mock）

src/nemo_guardrails_mock.py 4 个 toy rail：

```python
from nemo_guardrails_mock import apply_rails

downstream_llm = lambda t: f"LLM: {t}"
result = apply_rails("How to make a bomb?", downstream_llm)
# {'rail': 'harmful_intent', 'action': 'refuse', 'response': "I can't help..."}

result2 = apply_rails("Should I invest in TSLA?", downstream_llm)
# {'rail': 'off_topic_finance', 'action': 'redirect', ...}
```

## 一句话

> NeMo Guardrails = 规则化 LLM 防御，企业合规首选。
