# L12 · Multi-Agent 成本 — Token 爆炸

## Token 爆炸现实

```
Single agent:    1 prompt × N tokens = N tokens
3-agent debate × 3 rounds:
   3 agent × 3 rounds × N tokens × accumulated context
   ≈ 9 × N × 2 (每 agent 看 prev rounds)
   ≈ 18 × N (理想)
   ≈ 30 × N (现实有 system prompt + tool calls)
```

→ 30 倍成本只换 4-8 pp accuracy。是否值？

## 成本 / 收益 cheat sheet

| 任务 | 单 agent | Multi-agent ROI |
|------|---------|----------------|
| 简单 QA | $0.001 | ❌ |
| 数学（GSM8K） | $0.005 | ⚠️ 视精度需要 |
| 代码 (HumanEval) | $0.02 | ✓ 调试时间 > token 成本 |
| 研究报告 | $0.50 | ✓ 质量决定一切 |
| Web agent | $1.00 | ✓ 能跑通就赚 |
| 客服 chatbot | $0.0001 × 1000 user | ❌ 量大 |

## 何时绝对不用 multi-agent

| 反场景 | 原因 |
|--------|------|
| Per-request budget $0.01 | 1 LLM call 就花了 |
| Latency < 2s 硬要求 | 多轮慢 |
| 1k+ QPS | 成本爆炸 |
| Task 已经 95% acc | 边际收益小 |

## 工程优化技巧

### 1. Context pruning

```python
# 不让 supervisor 看完整 history
def summarize_for_supervisor(history):
    return llm(f"Summarize in 100 tokens: {history}")
```

### 2. Selective routing

```
Only run debate when:
  - confidence < 0.7
  - 关键决策
```

### 3. Worker model 降级

```
Supervisor: GPT-4o ($0.005/req)
Worker:     GPT-4o-mini ($0.0001/req)
→ 50× 便宜
```

### 4. Cache

```python
# 同样的 sub-task → cache
hash_key = hash(task_description)
if cached := cache.get(hash_key):
    return cached
```

### 5. Parallel

```
Sequential 3 agent: 3 × latency
Parallel 3 agent:   1 × latency (但还是 3× cost)
```

## 实现 (`cost_analyzer.py` 预告)

```python
@dataclass
class CostTracker:
    tokens_in: int = 0
    tokens_out: int = 0
    n_calls: int = 0

    def add_call(self, tokens_in, tokens_out):
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.n_calls += 1

    def usd(self, in_price=0.003, out_price=0.015):
        return (self.tokens_in / 1000 * in_price +
                self.tokens_out / 1000 * out_price)
```

## 2025 实战观察

Anthropic blog "Building effective agents" (2024.12):
> 大多数 production agent 1-2 个 LLM call 就够了。
> 多 agent 系统通常不值得。

→ 简单优先，复杂 multi-agent 是**最后选择**。

## 退出条件

- 知道 multi-agent 30× cost 数字
- 能列 5 优化技巧
- 知道 Anthropic 建议简单优先

## 一句话

> Multi-agent ≈ 30× 单 agent 成本 — 简单先做单 agent，必要时再上 multi，并用 model 降级 + cache。
