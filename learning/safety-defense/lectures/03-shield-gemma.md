# L03 · ShieldGemma (Google 2024)

## 数据

- 3 个尺寸：**2B / 9B / 27B**
- 基于 Gemma 2 fine-tune
- 训练数据：合成 + 人工标注
- Apache 2.0 开源

## 6 类 harm category

| 类 | 解释 |
|----|------|
| dangerous | 武器 / 暴力 / 恶意软件 |
| harassment | 针对个人或群体的骚扰 |
| hate | 仇恨言论 |
| sexual | 性内容 |
| self_harm | 自伤鼓励 |
| toxic | 一般毒性 |

比 Llama Guard 3 少（14→6），但**每类更细粒度**。

## 输出格式

```
Probability of harm: 0.92
```

返回 **continuous score**，不是 binary。

## 性能

| Model | Avg F1 |
|-------|--------|
| Llama Guard 3 8B | 0.85 |
| **ShieldGemma 9B** | **0.86** |
| ShieldGemma 27B | 0.88 |
| ShieldGemma 2B | 0.79 |

## 与 Llama Guard 3 对比

| 维度 | Llama Guard 3 | ShieldGemma |
|------|---------------|-------------|
| 大小 | 1B / 8B | 2B / 9B / 27B |
| 类别 | 14 (MLCommons) | 6 (自定义) |
| 输出 | binary + class | probability |
| 训练 | 合成主 | 合成 + 人工 |
| 平台 | Meta + HF | Google + HF + Vertex |

## 用法

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("google/shieldgemma-9b")
mdl = AutoModelForCausalLM.from_pretrained("google/shieldgemma-9b").cuda()

prompt = f"""<start_of_turn>user
{user_input}<end_of_turn>
Our safety principle is: ...
Probability of violation:"""

ids = tok(prompt, return_tensors="pt").input_ids.cuda()
out = mdl.generate(ids, max_new_tokens=5)
prob = float(tok.decode(out[0][ids.shape[1]:]).strip())
```

## 选择 guideline

| 场景 | 推荐 |
|------|------|
| Llama 生态 | Llama Guard 3 |
| Gemma / Vertex AI | ShieldGemma |
| 多语言 | ShieldGemma 27B（更广）|
| 紧凑边缘 | Llama Guard 1B |

## 一句话

> ShieldGemma = Google 版 Llama Guard，3 个尺寸 + 概率输出。
