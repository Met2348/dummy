# L04 · Chatbot Arena Elo — 真人盲对

**LMSYS** · chat.lmsys.org · arXiv 2403.04132

## 核心机制

```
1. 用户输入 prompt
2. 系统随机抽 2 个 model（盲）
3. 各生成 response
4. 用户选 A 好 / B 好 / tie / both_bad
5. 这场对决进入 leaderboard 计算
```

100 万+ 真人投票，**最权威的 LLM 排行**。

## Bradley-Terry → Elo

```
Step 1: 对所有 (model_a, model_b, winner) 三元组
Step 2: 最大化 likelihood 求 log strength r_i
Step 3: Elo = 1500 + (400 / ln 10) × r_i
```

每周 retrain，含 bootstrap 95% CI。

## 当前排行（2025.06）

| Rank | Model | Elo |
|------|-------|-----|
| 1 | o3 | 1442 |
| 2 | Claude 3.7 Sonnet (extended thinking) | 1432 |
| 3 | Gemini 2.5 Pro | 1421 |
| 4 | GPT-4o (2025-05) | 1409 |
| 5 | DeepSeek V3 | 1377 |
| ... | ... | ... |
| ~25 | Llama 3.1 70B | 1248 |

差 ~200 Elo ≈ 10:1 win ratio。

## 多语言 / 多类别 leaderboard

- Overall
- English
- Chinese
- Coding
- Math
- Long Query
- Hard Prompts (Arena-Hard)
- Multi-turn

不同切片排名差异大。

## Style control

2025 起所有排行用 **style-controlled Elo**：
- 控制 response length / markdown / emoji
- 排除 "longer always wins" 的偏差

## 复现 Elo 计算

```python
import math
def k_factor_elo(elo_a, elo_b, winner, k=4):
    p_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    s_a = 1.0 if winner == "A" else 0.5 if winner == "tie" else 0.0
    elo_a += k * (s_a - p_a)
    elo_b += k * ((1 - s_a) - (1 - p_a))
    return elo_a, elo_b
```

注：上式是 sequential Elo (国际象棋经典)。Chatbot Arena 用 **MLE BT**（一次全局拟合），结果更稳。

## 实操

src/bradley_terry.py：

```python
from bradley_terry import fit_bt, to_elo
from common import PairBattle

battles = [PairBattle("q","A","B","A")] * 10 + [PairBattle("q","B","C","A")] * 10
bt = fit_bt(battles)
print(to_elo(bt))  # {'A': ~1700, 'B': ~1500, 'C': ~1300}
```

## 一句话

> Chatbot Arena = LLM 世界的"FIDE 等级分"，真人盲对最权威。
