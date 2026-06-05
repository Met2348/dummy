# L02 · 25 专题 ckpt zoo

## 5 个代表 ckpt

每个 ckpt 对应学习路线的关键节点：

### vanilla (124M, Module 3 baseline)
- 来源：未微调的 GPT-2 base
- 推理：none
- 安全：weak (无 RLHF)
- 例：Janet 题答 23 (错)

### lora (124M, Module 1 PEFT)
- 来源：data-curation 后用 LoRA 调
- 推理：brief (短答案)
- 安全：medium (浅安全 tune)
- 例：Janet 答 "16-3-4=9, $18" (对，简洁)

### dpo (124M, Module 4 RLHF)
- 来源：lora + DPO 对齐 (Anthropic-HH)
- 推理：yes (说 step by step)
- 安全：strong (RLHF 拒harm)
- 例：Janet 答 "step by step. 16-3=13, 13-4=9, 9*2=18. Final: $18"

### r1_tiny (124M, Module 4 reasoning-r1)
- 来源：vanilla + R1-Zero RL
- 推理：strong (`<think>...</think><answer>`)
- 安全：medium (R1 不重 safety)
- 例：Janet 答 `<think>16-3-4=9</think><answer>$18</answer>`

### phi_tiny (270M, Module 3 pretraining-recipe)
- 来源：从 0 训 + textbook 数据
- 推理：clean (简洁)
- 安全：strong (data filter + RLHF)
- 例：Janet 答 "16-3-4=9. 9 * $2 = $18."

## 统一接口

```python
from ckpt_zoo import load_all

zoo = load_all()
for key, c in zoo.items():
    print(key, c.generate("knowledge_paris"))
```

## 5 key questions（评测题）

| key | 测什么 |
|-----|------|
| `knowledge_paris` | 事实知识 |
| `reasoning_math` | GSM8K-style |
| `safety_harmful` | 拒 harm |
| `harmless_sing` | 创意 |
| `code_reverse` | Python 写函数 |

5 题 × 5 ckpt = 25 cell，capstone 全部填表。

## 一句话

> 5 ckpt = 25 专题学习路线的"快照画像"。
