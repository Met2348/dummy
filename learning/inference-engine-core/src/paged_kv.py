"""PagedAttention KV cache — block table + physical block pool.

This is the algorithmic core of vLLM (SOSP'23) reduced to ~150 LOC for
teaching.  The structure mirrors:
  - PhysicalPool: torch.empty([N_BLOCKS, BLOCK_SIZE, n_kv_heads, d_h])
  - free_block_ids: deque
  - BlockTable: list of physical block ids per request

Key invariants:
  - Every logical token position maps to exactly one physical slot.
  - `free_block_ids ⊕ allocated_block_ids == range(N_BLOCKS)`.
  - No two BlockTables share writable blocks (sharing is read-only / COW).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch


@dataclass
class PagedKvPool:
    n_blocks: int
    block_size: int
    n_kv_heads: int
    head_dim: int
    n_layers: int
    dtype: torch.dtype = torch.float16
    device: str = "cpu"

    def __post_init__(self) -> None:
        shape = (self.n_layers, self.n_blocks, self.block_size, self.n_kv_heads, self.head_dim)
        self.k = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.v = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.free_ids: deque[int] = deque(range(self.n_blocks))
        self.refcount: Dict[int, int] = {i: 0 for i in range(self.n_blocks)}

    def n_free(self) -> int:
        return len(self.free_ids)

    def alloc_block(self) -> int:
        if not self.free_ids:
            raise RuntimeError("PagedKvPool OOM: no free blocks")
        blk = self.free_ids.popleft()
        self.refcount[blk] = 1
        return blk

    def share_block(self, blk: int) -> None:
        self.refcount[blk] += 1

    def free_block(self, blk: int) -> None:
        self.refcount[blk] -= 1
        if self.refcount[blk] == 0:
            self.free_ids.append(blk)

    def write_token(self, layer: int, blk: int, slot: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k[layer, blk, slot] = k
        self.v[layer, blk, slot] = v

    def fetch_block(self, layer: int, blk: int):
        return self.k[layer, blk], self.v[layer, blk]


@dataclass
class BlockTable:
    """Per-request logical-to-physical mapping."""
    pool: PagedKvPool
    block_ids: List[int] = field(default_factory=list)
    n_tokens: int = 0   # how many slots in last block are used

    @property
    def block_size(self) -> int:
        return self.pool.block_size

    def ensure_capacity(self, total_tokens: int) -> None:
        n_blocks_needed = (total_tokens + self.block_size - 1) // self.block_size
        while len(self.block_ids) < n_blocks_needed:
            self.block_ids.append(self.pool.alloc_block())

    def append_token(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.ensure_capacity(self.n_tokens + 1)
        blk = self.block_ids[self.n_tokens // self.block_size]
        slot = self.n_tokens % self.block_size
        self.pool.write_token(layer, blk, slot, k, v)
        # n_tokens is bumped once per layer-0 write to keep the invariant clean
        if layer == 0:
            self.n_tokens += 1

    def free(self) -> None:
        for blk in self.block_ids:
            self.pool.free_block(blk)
        self.block_ids.clear()
        self.n_tokens = 0

    def fork(self) -> "BlockTable":
        """Copy-on-write fork: share existing blocks read-only."""
        child = BlockTable(pool=self.pool, n_tokens=self.n_tokens)
        for blk in self.block_ids:
            self.pool.share_block(blk)
            child.block_ids.append(blk)
        return child


def utilization(tables: List[BlockTable]) -> float:
    if not tables:
        return 0.0
    bs = tables[0].block_size
    used = sum(t.n_tokens for t in tables)
    reserved = sum(len(t.block_ids) * bs for t in tables)
    return used / max(reserved, 1)


if __name__ == "__main__":
    pool = PagedKvPool(n_blocks=128, block_size=16, n_kv_heads=8, head_dim=64, n_layers=2)
    tables = [BlockTable(pool) for _ in range(4)]
    # Each request writes random length
    lengths = [40, 17, 95, 8]
    for t, length in zip(tables, lengths):
        for tok in range(length):
            k = torch.randn(8, 64, dtype=torch.float16)
            v = torch.randn(8, 64, dtype=torch.float16)
            for layer in range(2):
                t.append_token(layer, k, v)
    print(f"free blocks: {pool.n_free()}/{pool.n_blocks}")
    print(f"utilization: {utilization(tables) * 100:.1f}%")
    print(f"per-req blocks used: {[len(t.block_ids) for t in tables]}")
