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


def demo() -> None:
    """Show MoE expert->GPU placement, routing load imbalance, all-to-all cost."""
    moe = MoEEpDemo(n_experts=8, n_gpus=4, top_k=2)
    print("=== Expert Parallel (MoE) ===")
    print(f"experts={moe.n_experts}, gpus={moe.n_gpus}, top_k={moe.top_k}, "
          f"{moe.experts_per_gpu} experts/GPU\n")
    print("expert -> GPU placement:",
          {e: moe.assign_expert_to_gpu(e) for e in range(moe.n_experts)})

    rng = random.Random(0)
    routes = moe.route_tokens(list(range(4096)), rng)
    loads = moe.load_per_gpu(routes)
    print(f"\nrouting 4096 tokens (uniform): per-GPU load = {loads}")
    print(f"load imbalance (max/avg - 1) = {moe.load_imbalance(loads):.3f}  "
          f"(~0 means balanced; real routing is hot-spotty -> aux-loss-free balancing)")

    print("\n--- all-to-all dispatch time (ring model, each rank sends (N-1)/N) ---")
    print(f"{'n_ranks':>8}{'MB/rank':>9}{'NVLink ms':>12}{'PCIe ms':>10}")
    for n in (8, 16, 64):
        payload = 4_000_000
        nv = all_to_all_time_ms(n, payload, bw_gbps=900)
        pcie = all_to_all_time_ms(n, payload, bw_gbps=60)
        print(f"{n:>8}{payload // 1_000_000:>9}{nv:>12.3f}{pcie:>10.3f}")
    print("=> 2 all-to-all per MoE layer; PCIe is ~15x slower -> NVLink/NVSwitch needed.")


if __name__ == "__main__":
    demo()
