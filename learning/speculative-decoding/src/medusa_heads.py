"""Medusa multi-head draft (Cai 2024).

Mocks `M` heads, each predicting an offset distribution from the backbone
hidden state.  We don't ship trained weights; instead each head is a
permutation noise on the target distribution so the workshop runs without
checkpoints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List
import random

from common import softmax, sample_from, SpecMetrics


@dataclass
class MedusaHeads:
    n_heads: int = 4
    noise: float = 0.5    # how much each head's output differs from target

    def draft(self, target_dist: List[float], rng: random.Random) -> List[List[float]]:
        """Each head outputs a noisy version of `target_dist`."""
        out = []
        for h in range(self.n_heads):
            noisy = [max(p + rng.uniform(-self.noise, self.noise), 0.0) for p in target_dist]
            s = sum(noisy)
            out.append([p / s for p in noisy])
        return out


def medusa_step(
    target_fn: Callable[[List[int]], List[float]],
    heads: MedusaHeads,
    prefix: List[int],
    rng: random.Random,
) -> tuple[List[int], int]:
    """One Medusa iteration: draft `n_heads` then verify with target."""
    target_dist = target_fn(prefix)
    drafts = heads.draft(target_dist, rng)
    drafted_ids = [sample_from(d, rng) for d in drafts]

    # Verify: target gives p at positions 0..n_heads
    accepted: List[int] = []
    local = list(prefix)
    for i, draft in enumerate(drafted_ids):
        p = target_fn(local)
        q = drafts[i]
        r = rng.random()
        if r < p[draft] / max(q[draft], 1e-12):
            accepted.append(draft)
            local.append(draft)
        else:
            residual = [max(0.0, pj - qj) for pj, qj in zip(p, q)]
            s = sum(residual)
            tok = sample_from([r / s for r in residual] if s > 0 else p, rng)
            accepted.append(tok)
            return accepted, i + 1
    # bonus
    p_bonus = target_fn(local)
    accepted.append(sample_from(p_bonus, rng))
    return accepted, len(drafted_ids)
