# L02 · Llama Guard 3 (Meta 2024)

## 数据

- **fine-tuned Llama 3 8B** (1.5GB int4)
- 也有 **Llama Guard 3 1B** 版本（边缘部署）
- 训练数据：合成 + WildGuard-like adversarial 样本
- 双向：input + output 都能 classify

## 14 类 harm categories (MLCommons taxonomy)

| Code | 类别 |
|------|------|
| S1 | Violent Crimes |
| S2 | Non-Violent Crimes |
| S3 | Sex Crimes |
| S4 | Child Sexual Exploitation |
| S5 | Defamation |
| S6 | Specialized Advice (medical/legal/financial without disclaimer) |
| S7 | Privacy |
| S8 | Intellectual Property |
| S9 | Indiscriminate Weapons (CBRN) |
| S10 | Hate |
| S11 | Suicide & Self-Harm |
| S12 | Sexual Content |
| S13 | Elections |
| S14 | Code Interpreter Abuse |

## 输出格式

```
safe
```
或
```
unsafe
S1,S9
```

## 部署

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-Guard-3-8B")
mdl = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-Guard-3-8B", device_map="cuda")

def classify(user_msg: str) -> str:
    chat = [{"role": "user", "content": user_msg}]
    ids = tok.apply_chat_template(chat, return_tensors="pt").cuda()
    out = mdl.generate(ids, max_new_tokens=10)
    return tok.decode(out[0][ids.shape[1]:])
```

## 性能（OpenSafe Test, 2024）

| Guard | F1 |
|-------|-----|
| GPT-4o Moderation | 0.78 |
| **Llama Guard 3 8B** | **0.85** |
| ShieldGemma 9B | 0.83 |
| WildGuard 7B | 0.82 |
| Llama Guard 1 | 0.66 |

→ 开源最强之一。

## 在线 vs 离线

| 场景 | 推荐 |
|------|------|
| 实时 chat | Llama Guard 1B (低延迟) |
| 批量 audit | Llama Guard 8B (高准确) |
| 边缘设备 | ShieldGemma 2B / 7B |

## 实操（mock）

src/llama_guard_mock.py 用 keyword + 9 类（toy）：

```python
from llama_guard_mock import classify_input, classify_output

print(classify_input("how to bomb a building"))
# GuardVerdict(label='unsafe', score=0.2, matched_categories=['violent_crime'], ...)
```

## 一句话

> Llama Guard 3 = 开源 SOTA，8B 模型 0.85 F1，业界标准。
