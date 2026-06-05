# L05 · AlpacaEval 2 LC (length-controlled)

**Dubois et al. 2024** · arXiv 2404.04475

## 背景

AlpacaEval v1 用 win rate vs `text-davinci-003`。
**问题**：模型只要写得长，就 win rate 高（GPT-4 judge 偏好长答案）。

→ **length-controlled (LC) winrate**

## 公式

```
raw_winrate = mean(judge_says_model_wins)

# Fit logistic regression: P(win) ~ length_diff
α, β = logistic_fit(wins, length_diffs)
length_effect = sigmoid(α + β * length_diff)

lc_winrate = raw_winrate - (length_effect.mean() - 0.5)
```

直觉：**减去"长答案该赢的部分"**。

## 数据

- **805 instructions** 来自 AlpacaFarm
- baseline = **GPT-4-turbo-2024-04-09**
- judge = GPT-4-turbo（同模型，但提示工程精心）

## 分数（AlpacaEval 2 LC, 2024-2025）

| 模型 | LC win rate vs GPT-4-turbo |
|------|---------------------------|
| GPT-4-turbo | 50.0 |
| GPT-4o | 57.5 |
| Claude 3 Opus | 40.5 |
| Claude 3.5 Sonnet | 52.4 |
| **Claude 3.7 Sonnet** | **63.1** |
| Llama 3.1 70B-Instruct | 38.1 |

## v1 vs v2 LC 差异

| 模型 | v1 raw | v2 LC |
|------|--------|-------|
| Llama 3 8B | 22% | 22.9% |
| GPT-4 0314 | 22% | 34.5% |

→ v1 严重偏好长答案，v2 校正后排名变化大。

## LC 公式有限性

只校正 **平均 length 偏差**，不校正：
- markdown 偏好
- emoji 偏好
- "I'd be happy to help" 套话

→ Arena-Hard 的 style control 更全面。

## 实操

src/alpaca_eval.py：

```python
from alpaca_eval import lc_winrate
from common import PairBattle

# 'A' wins 60% but is much longer
battles = [PairBattle(f"q{i}", "A", "B", "A" if i < 6 else "B")
           for i in range(10)]
lengths = {f"q{i}": 500 for i in range(10)}  # A is 500 chars longer
print(lc_winrate(battles, lengths))
# lc_winrate < raw_winrate
```

## 一句话

> AlpacaEval 2 LC = 给 win rate 减去"长度税"，公平排名。
