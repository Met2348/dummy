"""Naive (static) KV cache — over-reserved [B, max_len] tensor.

Used as the baseline against which PagedAttention is measured.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import torch


@dataclass
class NaiveKvPool:
    """Static `[B, max_len, n_kv_heads, d_h]` KV cache.

    `actual_lens[b]` tracks the live (used) length for request b.
    """
    batch: int
    max_len: int
    n_kv_heads: int
    head_dim: int
    n_layers: int
    dtype: torch.dtype = torch.float16
    device: str = "cpu"

    def __post_init__(self) -> None:
        shape = (self.n_layers, self.batch, self.max_len, self.n_kv_heads, self.head_dim)
        self.k = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.v = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.actual_lens: List[int] = [0] * self.batch

    def write(self, layer: int, b: int, pos: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k[layer, b, pos] = k
        self.v[layer, b, pos] = v
        self.actual_lens[b] = max(self.actual_lens[b], pos + 1)

    def fetch(self, layer: int, b: int):
        n = self.actual_lens[b]
        return self.k[layer, b, :n], self.v[layer, b, :n]

    def memory_bytes(self) -> int:
        return 2 * self.k.numel() * self.k.element_size()

    def utilization(self) -> float:
        """Return fraction of reserved tokens actually populated."""
        used = sum(self.actual_lens)
        total = self.batch * self.max_len
        return used / max(total, 1)

    def reset(self, b: int) -> None:
        self.actual_lens[b] = 0


def demo_fragmentation(B: int = 8, max_len: int = 2048, avg_len: int = 512) -> dict:
    """Compute static-frag waste for a representative workload."""
    import random
    random.seed(0)
    lengths = [random.randint(avg_len // 2, avg_len * 3 // 2) for _ in range(B)]
    used_tokens = sum(lengths)
    reserved = B * max_len
    return {
        "batch": B,
        "max_len": max_len,
        "lengths": lengths,
        "used_tokens": used_tokens,
        "reserved": reserved,
        "utilization": used_tokens / reserved,
        "wasted_tokens": reserved - used_tokens,
        "wasted_pct": 1 - used_tokens / reserved,
    }


if __name__ == "__main__":
    print("Fragmentation demo:")
    info = demo_fragmentation()
    for k, v in info.items():
        print(f"  {k}: {v}")
    print(f"\n=> {info['wasted_pct'] * 100:.1f}% of KV slots wasted under naive layout")
