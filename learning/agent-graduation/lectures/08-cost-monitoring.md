# L08 · 成本监控

## 30 秒

> DRA 单 run $0.05-$0.5，10k user/day → $500-$5000/day → 必须监控、必须 cap。

## 4 大成本维

| 维 | 工具 | 价格 (2025) |
|---|------|------------|
| **LLM tokens** | Anthropic / OpenAI | $0.10-15 / 1M |
| **Embeddings** | OpenAI / Voyage | $0.02-0.13 / 1M |
| **Search API** | Brave / Tavily | $0.005 / req |
| **Sandbox exec** | e2b / Modal | $0.0005 / sec |

## 成本聚合 per run

```python
@dataclass
class RunCost:
    llm_in: int = 0
    llm_out: int = 0
    embed: int = 0
    search_calls: int = 0
    sandbox_seconds: float = 0

    def usd(self) -> float:
        return (self.llm_in / 1e6 * 3.0
                + self.llm_out / 1e6 * 15.0
                + self.embed / 1e6 * 0.13
                + self.search_calls * 0.005
                + self.sandbox_seconds * 0.0005)
```

## Per-user budget

```python
@dataclass
class UserBudget:
    daily_usd: float = 5.0
    monthly_usd: float = 50.0
    spent_today: float = 0
    spent_month: float = 0

    def check(self, cost: float) -> bool:
        return (self.spent_today + cost <= self.daily_usd
                and self.spent_month + cost <= self.monthly_usd)
```

## 实时监控

```
Stream cost → metrics:
  - cost_per_user_today
  - cost_per_run_p50/p95/p99
  - tokens_per_run histogram
  - failed_runs_cost

→ Prometheus / DataDog / LangSmith
```

## Alert 规则

| 规则 | Action |
|------|--------|
| User > 80% daily | warn |
| User > 100% daily | block + email |
| Org > 50% monthly | warn ops |
| Single run > $5 | log + investigate |
| Cost p99 > 3× normal | alarm |

## Cost-aware routing

```python
def select_model(complexity: str, cost_remaining: float) -> str:
    if cost_remaining < 0.1:
        return "haiku"  # cheap
    if complexity == "high":
        return "claude-opus"
    return "claude-sonnet"
```

## 反例：cost leak

| Leak | 解 |
|------|---|
| Infinite loop tool call | max_steps + cycle detect |
| 大文件 fetch | size cap |
| Unbounded sub-agent spawn | depth cap |
| Cache miss 风暴 | prompt cache 启 |
| Retry 雪崩 | exponential backoff cap |

## 退出条件

- 能列 4 cost 维
- 能写 RunCost class
- 能列 5 alert 规则

## 一句话

> DRA cost = LLM + embed + search + sandbox 4 维 — per-user budget + 实时监控 + alert + cost-aware routing 防 leak。
