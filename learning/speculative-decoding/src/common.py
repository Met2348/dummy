"""Shared utilities for speculative decoding experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List
import math
import random


def softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]


def sample_from(probs: List[float], rng: random.Random) -> int:
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r < cum:
            return i
    return len(probs) - 1


def log_softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps)
    return [math.log(e / s) for e in exps]


@dataclass
class SpecMetrics:
    n_iters: int = 0
    n_drafted: int = 0
    n_accepted: int = 0
    n_tokens_out: int = 0

    @property
    def accept_rate(self) -> float:
        return self.n_accepted / max(self.n_drafted, 1)

    @property
    def mau(self) -> float:
        """Mean accepted per iter (excluding bonus)."""
        return self.n_accepted / max(self.n_iters, 1)
