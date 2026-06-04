"""Lookahead Decoding (UCSD 2024) — Jacobi-style + n-gram pool."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple


@dataclass
class NgramPool:
    """Maps prefix -> next-token id seen recently."""
    pool: Dict[Tuple[int, ...], int] = field(default_factory=dict)
    n: int = 3

    def add_sequence(self, seq: List[int]) -> None:
        for i in range(len(seq) - self.n):
            self.pool[tuple(seq[i : i + self.n])] = seq[i + self.n]

    def lookup(self, prefix: List[int]) -> int | None:
        if len(prefix) < self.n:
            return None
        return self.pool.get(tuple(prefix[-self.n :]))


def lookahead_step(
    target_fn: Callable[[List[int]], int],
    prefix: List[int],
    pool: NgramPool,
    lookahead_branches: int = 4,
) -> List[int]:
    """Generate next token AND lookahead branches; verify if any matches pool."""
    accepted = [target_fn(prefix)]
    # try to extend via pool
    local = prefix + accepted
    for _ in range(lookahead_branches):
        nxt = pool.lookup(local)
        if nxt is None:
            break
        accepted.append(nxt)
        local.append(nxt)
    pool.add_sequence(prefix + accepted)
    return accepted
