"""Prompt cache simulator (Anthropic / OpenAI style)."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from common import mock_now


@dataclass
class CacheEntry:
    timestamp: float
    response: str
    n_hits: int = 0
    tokens: int = 0


class PromptCache:
    def __init__(self, ttl_seconds: float = 300.0, min_prefix_tokens: int = 1024):
        self.cache: dict[str, CacheEntry] = {}
        self.ttl = ttl_seconds
        self.min_prefix_tokens = min_prefix_tokens
        self.stats = {"writes": 0, "hits": 0, "misses": 0}

    def _key(self, prefix: str) -> str:
        return hashlib.sha256(prefix.encode("utf-8")).hexdigest()

    def get(self, prefix: str) -> str | None:
        key = self._key(prefix)
        if key not in self.cache:
            self.stats["misses"] += 1
            return None
        entry = self.cache[key]
        if mock_now() - entry.timestamp > self.ttl:
            del self.cache[key]
            self.stats["misses"] += 1
            return None
        entry.n_hits += 1
        self.stats["hits"] += 1
        return entry.response

    def put(self, prefix: str, response: str, tokens: int = 0) -> bool:
        if tokens < self.min_prefix_tokens:
            return False
        self.cache[self._key(prefix)] = CacheEntry(
            timestamp=mock_now(), response=response, tokens=tokens,
        )
        self.stats["writes"] += 1
        return True

    def cost_estimate(
        self,
        n_calls: int,
        prefix_tokens: int,
        write_price: float = 3.75,
        cached_read_price: float = 0.30,
        regular_price: float = 3.00,
    ) -> dict:
        per_million = 1_000_000.0
        no_cache = n_calls * prefix_tokens / per_million * regular_price
        cache_cost = (prefix_tokens / per_million * write_price
                      + (n_calls - 1) * prefix_tokens / per_million * cached_read_price)
        savings = no_cache - cache_cost
        return {
            "no_cache_usd": round(no_cache, 4),
            "cache_usd": round(cache_cost, 4),
            "savings_usd": round(savings, 4),
            "savings_pct": round(100 * savings / max(1e-9, no_cache), 1),
        }


def _self_test() -> None:
    from common import reset_mock_time

    reset_mock_time()
    cache = PromptCache(ttl_seconds=10.0, min_prefix_tokens=100)
    prefix = "long system prompt " * 200
    assert cache.put(prefix, "response v1", tokens=2000) is True
    assert cache.put("short", "x", tokens=10) is False

    assert cache.get(prefix) == "response v1"
    assert cache.stats["hits"] == 1

    assert cache.get("nothing") is None
    assert cache.stats["misses"] == 1

    estimate = cache.cost_estimate(n_calls=100, prefix_tokens=10000)
    assert estimate["savings_pct"] > 80, estimate
    print(f"[OK] prompt_cache._self_test passed (savings {estimate['savings_pct']}%)")


if __name__ == "__main__":
    _self_test()
