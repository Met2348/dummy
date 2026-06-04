"""EAGLE — feature-level autoregressive draft (Li 2024)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List
import random

from common import softmax, sample_from, SpecMetrics


@dataclass
class EagleDraft:
    """Single transformer-layer draft running auto-regressively over `k` steps."""
    k: int = 4
    noise: float = 0.30   # quality knob — EAGLE-style closer to target than Medusa

    def draft_sequence(
        self,
        target_dist_fn: Callable[[List[int]], List[float]],
        prefix: List[int],
        rng: random.Random,
    ) -> tuple[List[int], List[List[float]]]:
        """Auto-regressively draft `k` tokens; return (ids, per-step distributions)."""
        local = list(prefix)
        ids: List[int] = []
        qs: List[List[float]] = []
        for _ in range(self.k):
            true_p = target_dist_fn(local)
            noisy = [max(p + rng.uniform(-self.noise, self.noise), 0.0) for p in true_p]
            s = sum(noisy)
            q = [p / s for p in noisy]
            tid = sample_from(q, rng)
            ids.append(tid)
            qs.append(q)
            local.append(tid)
        return ids, qs


def eagle_step(
    target_fn: Callable[[List[int]], List[float]],
    draft: EagleDraft,
    prefix: List[int],
    rng: random.Random,
) -> tuple[List[int], int]:
    drafted_ids, qs = draft.draft_sequence(target_fn, prefix, rng)
    accepted: List[int] = []
    local = list(prefix)
    for i, tid in enumerate(drafted_ids):
        p = target_fn(local)
        r = rng.random()
        if r < p[tid] / max(qs[i][tid], 1e-12):
            accepted.append(tid)
            local.append(tid)
        else:
            residual = [max(0.0, pj - qj) for pj, qj in zip(p, qs[i])]
            s = sum(residual)
            tok = sample_from([r / s for r in residual] if s > 0 else p, rng)
            accepted.append(tok)
            return accepted, i + 1
    p_bonus = target_fn(local)
    accepted.append(sample_from(p_bonus, rng))
    return accepted, len(drafted_ids)
