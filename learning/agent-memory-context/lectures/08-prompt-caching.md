# L08 · Prompt Caching

## 30 秒

> 长 system prompt + tool schemas + RAG context 每 turn 重传 token → API 成本爆炸。
>
> **Prompt cache** 让 provider 把 prefix 缓存，下次同 prefix 复用，**减 90% 输入 token 成本**。

## 时间线

| 时间 | 事件 |
|------|------|
| 2024.04 | Anthropic 推 prompt caching beta |
| 2024.10 | Anthropic GA |
| 2024.10 | OpenAI 推 prompt caching |
| 2024.12 | Google Gemini prompt cache |

## Anthropic prompt caching

```python
response = client.messages.create(
    model="claude-sonnet-4",
    system=[
        {"type": "text", "text": "System instructions...",
         "cache_control": {"type": "ephemeral"}}  # mark this for caching
    ],
    messages=[...]
)
```

- TTL: 5 分钟 (default) / 1 小时 (extended)
- Min prefix: 1024 token
- Up to 4 cache breakpoints per request

## OpenAI prompt caching

- 自动，不需 explicit `cache_control`
- 长 prompt 自动用前 1024 token 起 hash
- ChatGPT 后台对所有 enterprise / API 默认开

## 价格对照（2025 Anthropic）

| 操作 | 标准 / 1M tok | Cached | 倍 |
|------|--------------:|-------:|---:|
| Cache write | $3.75 (1.25×) | — | — |
| Read cached | $0.30 (0.1×) | — | — |
| 标准 input | $3.00 | — | — |

→ Cached read 是原价的 **1/10**！长 prompt + 多 turn 直接省 90%。

## 缓存命中策略

```
✓ System prompt → 始终 cache
✓ Tool schemas → 始终 cache
✓ Stable retrieved context → cache (改了就 invalidate)
✗ User message (随便变) → 不 cache
```

## Cache breakpoint

```python
system=[
    {"type":"text","text":"You are an agent...","cache_control":{"type":"ephemeral"}},
    {"type":"text","text":"<tools>...</tools>","cache_control":{"type":"ephemeral"}},
    {"type":"text","text":"<docs>...</docs>","cache_control":{"type":"ephemeral"}},
]
```

→ 多个 breakpoint，最后一个 hit 之前的 prefix 都生效。

## 实现 (`prompt_cache.py` 预告)

```python
class PromptCache:
    def __init__(self, ttl=300):
        self.cache = {}  # hash → (timestamp, value, n_hits)
        self.ttl = ttl

    def key(self, prefix: str) -> str:
        import hashlib
        return hashlib.sha256(prefix.encode()).hexdigest()

    def get(self, prefix):
        k = self.key(prefix)
        if k in self.cache:
            ts, val, hits = self.cache[k]
            if time.time() - ts < self.ttl:
                self.cache[k] = (ts, val, hits + 1)
                return val
        return None

    def put(self, prefix, value):
        self.cache[self.key(prefix)] = (time.time(), value, 0)
```

## 实战节省例

```
System (10k tok) + tools (5k) + RAG (20k) = 35k 前缀
× 100 turn 对话

不 cache: 35k × $3/M × 100 = $10.50
Cache:    35k × ($3.75 写 + 99×$0.30 读)/M = $1.32

省 87%
```

## 退出条件

- 能默写 Anthropic cache_control 用法
- 能讲 5min / 1h TTL
- 知道 cached read 是原价 1/10

## 一句话

> Prompt caching = 长 prefix hash + provider 缓存 — 长对话省 90% 输入成本，2024-2025 必用。
