# L11 · 冲突 Resolution

## 当 N agent 答案不一致

3 种解：

| 策略 | 实现 | 适用 |
|------|------|------|
| **Voting** | majority 投票 | 多选 / 二元 |
| **Weighted** | trust × vote | agent 可信度有差 |
| **Judge** | 第三 agent 当裁判 | 复杂答 |

## Majority voting

```python
def majority_vote(answers):
    counter = Counter(answers)
    return counter.most_common(1)[0][0]
```

简单暴力，2-of-3 / 3-of-5 多数即定。

## Weighted voting

```python
def weighted_vote(answers, weights):
    scores = defaultdict(float)
    for a, w in zip(answers, weights):
        scores[a] += w
    return max(scores, key=scores.get)
```

weights 可以来自：
- 历史准确率
- 模型大小 (GPT-4 > GPT-3.5)
- 任务匹配度

## Judge agent

```python
JUDGE_PROMPT = """
Question: {q}
Candidate answers:
{candidates}
Rate each on:
- correctness
- reasoning quality
- citations
Pick the best, explain why.
"""
```

LLM-as-judge — 但要小心 self-preference bias (Module 6 讲过)。

## Self-consistency vs Multi-agent vote

```
Self-consistency: 1 agent × N samples → vote
Multi-agent:     N agents × 1 sample → vote
```

数字（GSM8K, GPT-3.5）：

| Method | acc |
|--------|----:|
| Single | 77% |
| Self-consistency N=8 | 82% |
| Multi-agent N=3 | 81% |
| Multi-agent debate 3 round | 85% |

→ Self-consistency 跟 multi-agent vote 差不多，debate（critique）才显著强。

## 投票失效 case

| 反例 | 解 |
|------|---|
| 全错（相同失败模式） | diverse agents (prompt / model) |
| 平票 | tie-breaker / abstain |
| 多答（数字稍异） | answer canonicalization |
| 长输出 | 抽 final answer 再 vote |

## Tie-breaker

```python
def vote_with_tie_break(answers, fallback):
    counts = Counter(answers)
    top, count = counts.most_common(1)[0]
    if count == 1:
        return fallback   # no consensus
    return top
```

## 实现 (`conflict_resolution.py` 预告)

```python
def majority_vote(answers): ...
def weighted_vote(answers, weights): ...
def judge_pick(question, answers, judge_fn): ...
def borda_count(rankings): ...  # 排序投票
```

## 退出条件

- 能列 3 策略
- 知道 self-consistency vs multi-agent vote
- 能写 tie-breaker

## 一句话

> 冲突解 3 法 (vote / weighted / judge) — diverse 比 redundant 更值钱，平票需 tie-breaker。
