# L10 · Pairwise vs Pointwise — 决策树

## 公式回顾

**Pointwise**:
```
score = judge(question, response)  # ∈ [1, 5] or [1, 10]
model_score = mean(scores)
```

**Pairwise**:
```
winner = judge(question, resp_A, resp_B)  # 'A' / 'B' / 'tie'
win_rate = (#A + 0.5 * #tie) / total
```

## 优劣对比

| 维度 | Pointwise | Pairwise |
|------|-----------|----------|
| **成本** | O(N) | O(N²) per pair |
| **稳定性** | 分数漂移 | 相对稳 |
| **校准** | 难（分数标定）| 自然校准 |
| **Position bias** | 无 | 有 |
| **Length bias** | 重 | 中 |
| **小差异敏感** | 低 | 高 |

## 何时用 pointwise

- 单 model 评估（不需对比）
- 大规模数据集（10k+ samples）
- 想知道"绝对分数"

例子：MT-Bench、Prometheus 2、G-Eval。

## 何时用 pairwise

- 模型对比 / 排名
- 数据少（< 1000）
- 希望抗 noise

例子：Arena-Hard、Chatbot Arena、AlpacaEval。

## 混合策略

**两步法**（生产推荐）：
```
1. Pointwise 粗筛：100k → 1k 高质量
2. Pairwise 排名：1k → final ranking
```

省成本 + 抗 bias。

## 一致性研究

(Zheng et al. 2023, MT-Bench paper)
- pointwise 同一 response 两次评分：Cohen's κ = 0.45 (中等)
- pairwise 同一对两次评判：κ = 0.62 (好)

→ **pairwise 比 pointwise 稳定 ~30%**。

## listwise = pairwise 进化

```
listwise: judge(q, [r1, r2, r3, r4]) → ranking
```

- 优势：一次评 N 个
- 限制：LLM 处理 5+ 长 response 易混乱
- 实际：N ≤ 4 时效果接近 pairwise

## 我们的代码

src/mt_bench_runner.py: pointwise 实现
src/arena_hard_runner.py: pairwise 实现
src/mini_arena.py: round-robin 多模型 pairwise + BT

## 一句话

> Pointwise 便宜但漂，pairwise 稳但贵 — 混合策略最实用。
