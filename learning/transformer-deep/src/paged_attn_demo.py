"""PagedAttention 教学版 — block table + KV pool.

教学目标：
    1. block table 数据结构
    2. KV pool 分配 / 释放
    3. 解码时按 block 读 K, V

不实际跑 attention，只演示管理逻辑。
"""
from __future__ import annotations

from typing import Optional

import torch


class KVPool:
    def __init__(self, n_blocks: int, block_size: int, h_kv: int, d_head: int,
                 device="cpu", dtype=torch.float16):
        self.block_size = block_size
        self.h_kv = h_kv
        self.d_head = d_head
        # 单层 demo: shape (n_blocks, 2, block_size, h_kv, d_head)
        self.pool = torch.zeros(n_blocks, 2, block_size, h_kv, d_head,
                                device=device, dtype=dtype)
        self.free: list[int] = list(range(n_blocks))
        self.ref_count: dict[int, int] = {}

    def alloc(self) -> int:
        if not self.free:
            raise RuntimeError("KV pool exhausted")
        b = self.free.pop(0)
        self.ref_count[b] = 1
        return b

    def share(self, b: int) -> None:
        self.ref_count[b] = self.ref_count.get(b, 0) + 1

    def release(self, b: int) -> None:
        self.ref_count[b] -= 1
        if self.ref_count[b] == 0:
            self.free.append(b)
            del self.ref_count[b]

    @property
    def n_free(self) -> int:
        return len(self.free)


class Sequence:
    def __init__(self, pool: KVPool, seq_id: str):
        self.pool = pool
        self.seq_id = seq_id
        self.block_table: list[int] = []
        self.length = 0

    def append_token(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """k, v: shape (h_kv, d_head) — 一个 token."""
        if self.length % self.pool.block_size == 0:
            self.block_table.append(self.pool.alloc())
        block_id = self.block_table[-1]
        slot = self.length % self.pool.block_size
        self.pool.pool[block_id, 0, slot] = k
        self.pool.pool[block_id, 1, slot] = v
        self.length += 1

    def share_prefix_from(self, other: "Sequence", n_prefix: int) -> None:
        """共享前 n_prefix 个 token 对应的 block."""
        n_blocks = n_prefix // self.pool.block_size
        for i in range(n_blocks):
            b = other.block_table[i]
            self.block_table.append(b)
            self.pool.share(b)
        self.length = n_blocks * self.pool.block_size

    def free(self) -> None:
        for b in self.block_table:
            self.pool.release(b)
        self.block_table.clear()
        self.length = 0


def run_demo() -> None:
    pool = KVPool(n_blocks=20, block_size=4, h_kv=2, d_head=8)
    print(f"init free: {pool.n_free}")

    # seq 0: 写 10 token
    s0 = Sequence(pool, "s0")
    for _ in range(10):
        k = torch.randn(2, 8); v = torch.randn(2, 8)
        s0.append_token(k, v)
    print(f"after seq 0 (10 tokens): blocks={s0.block_table}, free={pool.n_free}")

    # seq 1: 共享 seq 0 的前 8 token (system prompt)
    s1 = Sequence(pool, "s1")
    s1.share_prefix_from(s0, n_prefix=8)
    for _ in range(6):
        s1.append_token(torch.randn(2, 8), torch.randn(2, 8))
    print(f"after seq 1 (shared 8 + new 6): blocks={s1.block_table}, free={pool.n_free}")
    print(f"  ref count: {pool.ref_count}")

    # seq 0 done
    s0.free()
    print(f"after seq 0 free: free={pool.n_free}, ref={pool.ref_count}")

    # seq 1 done
    s1.free()
    print(f"after seq 1 free: free={pool.n_free}")


if __name__ == "__main__":
    run_demo()
