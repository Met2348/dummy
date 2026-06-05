# L10 · Tool Error 重试

## Tool 失败 3 类

| 类 | 例 | 重试 |
|---|---|------|
| 瞬时 | network timeout / 5xx | ✓ exponential backoff |
| 半永久 | rate limit 429 | ✓ wait per Retry-After |
| 永久 | bad args / 4xx | ✗ 不重试 (回 LLM 重写 args) |

## Exponential backoff

```python
def retry(fn, max_attempts=4, base=0.5):
    for attempt in range(max_attempts):
        try:
            return fn()
        except TransientError:
            if attempt == max_attempts - 1: raise
            delay = base * (2 ** attempt) + random.uniform(0, 0.1)
            time.sleep(delay)
```

间隔：0.5s → 1s → 2s → 4s（带 jitter 防 thunder herd）。

## Circuit breaker

```python
class CircuitBreaker:
    def __init__(self, threshold=5, timeout=60):
        self.failures = 0
        self.opened_at = None

    def call(self, fn):
        if self.opened_at and time.time() - self.opened_at < self.timeout:
            raise CircuitOpenError()
        try:
            result = fn()
            self.failures = 0
            return result
        except Exception:
            self.failures += 1
            if self.failures >= self.threshold:
                self.opened_at = time.time()
            raise
```

3 状态：closed (正常) → open (拒绝) → half-open (试探)。

## Permanent error 回 LLM

```python
{
  "tool_result": "ERROR: city='NyC' not found. Did you mean 'NYC'?",
  "is_error": true
}
```

→ LLM 看到 error，自己改 args 重试。

## 实现 (`tool_retry.py` 预告)

```python
def with_retry(func, max_attempts=4, base=0.5):
    def wrapper(args):
        last = None
        for attempt in range(max_attempts):
            result = func(args)
            if result.ok:
                return result
            last = result
            if "permanent" in (result.error or "").lower():
                return result
            time.sleep(base * (2 ** attempt))
        return last
    return wrapper
```

## Fallback chain

```python
def search_with_fallback(q):
    return (
        google_search(q) or
        bing_search(q) or
        ddg_search(q) or
        cached_search(q)
    )
```

→ 多家 backup，任一成功即返。

## 何时不重试

| 场景 | 不重试 |
|------|-------|
| Bad args (4xx) | LLM 改 |
| Permission denied | 让 user 处理 |
| Idempotency 不安全 | 怕重复执行 (转账) |

## 重试可观察

```python
@trace_attribute("retry_attempt")
def search_with_retry(q):
    ...
```

→ Logged 在 trace 里，方便 debug。

## 退出条件

- 能讲 3 类失败
- 能写 exponential backoff with jitter
- 知道 circuit breaker 3 状态

## 一句话

> Tool 失败 3 类 (瞬时/限流/永久) — 前两类 backoff + circuit breaker，永久回 LLM 改 args。
