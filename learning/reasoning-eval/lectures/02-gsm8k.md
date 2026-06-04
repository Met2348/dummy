# L02 · GSM8K — grade school math

**Cobbe et al. 2021** · OpenAI · arXiv 2110.14168

## 数据

- **7473 train + 1319 test**（注：实际 8500 = 7.5k+1k 全文档）
- 多步算术，每题 2-8 步
- 答案 = 整数或简单小数
- 格式：自然语言解题 + `#### N` 末行

## 例题

```
Q: Janet's ducks lay 16 eggs/day. She eats 3 for breakfast and bakes
muffins for her friends with 4. She sells the remainder at the farmer's
market for $2 per egg. How much does she make every day?

A: She eats 3 + 4 = <<3+4=7>>7 eggs.
She sells 16 - 7 = <<16-7=9>>9 eggs.
She makes 9 × 2 = <<9*2=18>>18.
#### 18
```

`<<a=b>>` 是 calculator annotation（训练时可用 trace）。

## 评测协议

1. **few-shot CoT** 标准 (k=8)
2. **answer extract**: `#### N` 优先，fallback 取最后整数
3. **maj@N**: 当年 standard，因 greedy 太不稳定

## 经典 baseline 分数

| 模型 | GSM8K (pass@1) |
|------|---------------|
| GPT-3 (175B) | 17.7% (CoT) |
| LLaMA-2 70B | 56.8% |
| GPT-4 | 92.0% |
| Claude 3.7 | 96.5% |
| R1 | 89.5%（更难的 maj@64）|
| 人类 (小学高年级) | 90%+ |

## 历史地位

- 2021 发布 → 立刻成为数学推理金标准
- 2022-2023 CoT 革命的主要 demonstrator
- 2024 起被认为 **"几乎饱和"** → 转向 MATH/AIME

## 已知 bug

- ~1-2% 题目答案错（社区 audit）
- 部分题目歧义（"if the remainder is X"）

## 实操

src/gsm8k_runner.py 6 题 micro：
```python
from gsm8k_runner import run_gsm8k
from common import make_dummy_model

rs = run_gsm8k(make_dummy_model("0"))
# baseline acc = 0
```

## 一句话

> GSM8K 是推理 bench 的"圣经"，每个 LLM 都得给个分数。
