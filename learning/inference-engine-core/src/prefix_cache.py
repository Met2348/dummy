"""Prefix caching — flat block-hash variant.

The teaching version hashes each block of token ids; if the hash already lives
in the cache we mount the existing physical block on the new request instead
of recomputing the prefix.

Real engines layer this on top of `PagedKvPool`; here we abstract the pool
into a dictionary keyed by block hash.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import hashlib


def block_hash(tokens: Tuple[int, ...]) -> str:
    return hashlib.sha1(bytes(repr(tokens), "utf-8")).hexdigest()[:16]


@dataclass
class PrefixCache:
    block_size: int = 16
    cap: int = 256
    table: Dict[str, int] = field(default_factory=dict)     # hash -> phys_id
    next_phys: int = 0
    hits: int = 0
    misses: int = 0
    lru: List[str] = field(default_factory=list)

    def mount(self, token_ids: List[int]) -> Tuple[List[int], int]:
        """Return (phys_block_ids, hit_token_count) for a prompt prefix."""
        ids: List[int] = []
        hit_tokens = 0
        for i in range(0, len(token_ids), self.block_size):
            blk = tuple(token_ids[i : i + self.block_size])
            if len(blk) < self.block_size:
                break       # partial trailing block never cached
            h = block_hash(blk)
            if h in self.table:
                phys = self.table[h]
                self.hits += 1
                hit_tokens += self.block_size
                if h in self.lru:
                    self.lru.remove(h)
                self.lru.append(h)
            else:
                if len(self.table) >= self.cap:
                    evict = self.lru.pop(0)
                    self.table.pop(evict, None)
                phys = self.next_phys
                self.next_phys += 1
                self.table[h] = phys
                self.lru.append(h)
                self.misses += 1
            ids.append(phys)
        return ids, hit_tokens

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / max(total, 1)


if __name__ == "__main__":
    cache = PrefixCache(block_size=4, cap=8)
    sys_prompt = list(range(20))    # 5 blocks worth of system prompt
    for user_q in range(10):
        prompt = sys_prompt + [100 + user_q]
        phys_ids, hit_tokens = cache.mount(prompt)
        print(f"q{user_q}: phys={phys_ids} hit_tokens={hit_tokens}")
    print(f"hit_rate={cache.hit_rate:.3f}")
