"""Expert Parallel mock + load balance + all-to-all timing estimator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math
import random


@dataclass
class MoEEpDemo:
    n_experts: int = 8
    n_gpus: int = 4
    top_k: int = 2

    @property
    def experts_per_gpu(self) -> int:
        assert self.n_experts % self.n_gpus == 0
        return self.n_experts // self.n_gpus

    def assign_expert_to_gpu(self, expert_id: int) -> int:
        return expert_id // self.experts_per_gpu

    def route_tokens(self, tokens: List[int], rng: random.Random) -> List[Tuple[int, List[int]]]:
        """Returns list of (token_idx, [expert_ids_top_k])."""
        out = []
        for t in tokens:
            picks = rng.sample(range(self.n_experts), k=self.top_k)
            out.append((t, picks))
        return out

    def load_per_gpu(self, routes: List[Tuple[int, List[int]]]) -> List[int]:
        loads = [0] * self.n_gpus
        for _, experts in routes:
            for e in experts:
                loads[self.assign_expert_to_gpu(e)] += 1
        return loads

    def load_imbalance(self, loads: List[int]) -> float:
        avg = sum(loads) / max(len(loads), 1)
        return max(loads) / max(avg, 1e-9) - 1.0


def all_to_all_time_ms(n_ranks: int, bytes_per_rank: int, bw_gbps: float = 900.0) -> float:
    """Ring all-to-all time estimate: each rank sends (N-1)/N total bytes."""
    payload = bytes_per_rank * (n_ranks - 1) / n_ranks
    return payload / (bw_gbps * 1e9) * 1000.0
