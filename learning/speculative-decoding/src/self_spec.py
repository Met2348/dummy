"""Self-Speculative Decoding stub (skip-layer draft).

Mocks: target_fn requires "full" forward; skip_fn returns a noisier draft.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List
import random
from common import sample_from


@dataclass
class SelfSpec:
    skip_layers: int = 16
    base_noise: float = 0.5

    def draft(
        self,
        target_dist_fn: Callable[[List[int]], List[float]],
        prefix: List[int],
        rng: random.Random,
    ) -> tuple[int, List[float]]:
        true_p = target_dist_fn(prefix)
        # more skipped layers => higher noise
        noise = self.base_noise * self.skip_layers / 32
        noisy = [max(p + rng.uniform(-noise, noise), 0.0) for p in true_p]
        s = sum(noisy)
        q = [p / s for p in noisy]
        return sample_from(q, rng), q
