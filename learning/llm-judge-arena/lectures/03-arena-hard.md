# L03 · Arena-Hard (LMSYS 2024)

## 数据

- **500 hard prompts**：从 Chatbot Arena 实际数据中筛选
- 入选标准：模型间 win rate 方差大（"分得开"）
- vs 基线模型（**GPT-4-0314**）做 pairwise

## 评测

```
For each prompt:
  resp_A = candidate(prompt)
  resp_B = GPT-4-0314(prompt)
  winner = GPT-4-turbo_judge(prompt, resp_A, resp_B)
  # randomly swap A/B to detect position bias
```

Score = win rate vs baseline。

## 与 MT-Bench 差异

| 维度 | MT-Bench | Arena-Hard |
|------|---------|-----------|
| 题数 | 80 | 500 |
| 来源 | 人工设计 | Arena 真实 prompt |
| 难度 | 中 | 难（高方差）|
| 评测 | pointwise 1-10 | pairwise vs GPT-4 |
| 区分度 | 已饱和 | 仍很好 |

## 分数 (2024-2025)

| 模型 | Arena-Hard win rate |
|------|---------------------|
| GPT-4-0314 | 50% (baseline) |
| GPT-4o | 78.5% |
| Claude 3.5 Sonnet | 79.3% |
| Claude 3.7 Sonnet | 85.6% |
| Llama 3.1 405B | 64.1% |
| Qwen 2.5 72B | 65.8% |

## Bradley-Terry 升级

Arena-Hard 用 **bootstrap BT**：
1. resample battles (with replacement) 1000 次
2. 每次 fit BT
3. 报 mean + 95% CI

→ "GPT-4o 78.5% [95% CI: 77.1, 80.0]"

## Style control（2025）

LMSYS 后来加 **style-control regression**：
- 控制 response length / markdown 使用 / emoji
- 消除"长答案胜率"假相

→ Arena-Hard v2 / "Arena-Hard-Auto"

## 实操

src/arena_hard_runner.py：

```python
from arena_hard_runner import run_pairwise, win_rate
from common import make_fixed_gen, make_length_judge

a = make_fixed_gen({"mt_1": "longer answer"})
b = make_fixed_gen({"mt_1": "x"})
judge = make_length_judge(prefer_longer=True)
battles = run_pairwise(a, "A", b, "B", judge)
print(win_rate(battles, "A"))  # 1.0
```

## 一句话

> Arena-Hard = "现代版 MT-Bench"，500 真实难题 + pairwise + 防 length bias。
