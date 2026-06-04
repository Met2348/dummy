"""PagedAttention 教学模拟 - block 管理 + 逻辑/物理映射."""
from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Optional


BLOCK_SIZE = 16


@dataclass
class KvBlock:
    physical_id: int
    n_used: int = 0
    refcount: int = 1


class BlockManager:
    def __init__(self, total_blocks: int = 1024):
        self.total = total_blocks
        self.free = list(range(total_blocks))
        self.blocks: dict[int, KvBlock] = {}

    def alloc(self) -> Optional[KvBlock]:
        if not self.free:
            return None
        pid = self.free.pop(0)
        b = KvBlock(physical_id=pid, n_used=0, refcount=1)
        self.blocks[pid] = b
        return b

    def free_block(self, pid: int) -> None:
        b = self.blocks[pid]
        b.refcount -= 1
        if b.refcount == 0:
            del self.blocks[pid]
            self.free.append(pid)

    def share(self, pid: int) -> None:
        self.blocks[pid].refcount += 1

    def utilization(self) -> float:
        return (self.total - len(self.free)) / self.total


@dataclass
class Sequence:
    seq_id: int
    logical_blocks: list[int] = field(default_factory=list)

    def append_token(self, mgr: BlockManager):
        if not self.logical_blocks or mgr.blocks[
                self.logical_blocks[-1]].n_used >= BLOCK_SIZE:
            new = mgr.alloc()
            if new is None:
                raise RuntimeError("OOM")
            self.logical_blocks.append(new.physical_id)
        mgr.blocks[self.logical_blocks[-1]].n_used += 1


def fork_for_beam_search(parent: Sequence, mgr: BlockManager,
                         new_seq_id: int) -> Sequence:
    """CoW: 子序列共享父 blocks (refcount++)."""
    child = Sequence(seq_id=new_seq_id,
                     logical_blocks=list(parent.logical_blocks))
    for pid in child.logical_blocks:
        mgr.share(pid)
    return child


if __name__ == "__main__":
    mgr = BlockManager(total_blocks=64)
    print(f"Initial: util={mgr.utilization():.1%}")

    seq = Sequence(seq_id=0)
    for _ in range(50):
        seq.append_token(mgr)
    print(f"\nAfter 50 tokens: blocks={len(seq.logical_blocks)} "
          f"util={mgr.utilization():.1%}")

    fork = fork_for_beam_search(seq, mgr, new_seq_id=1)
    print(f"After fork: util={mgr.utilization():.1%} (shared, no alloc)")

    for _ in range(20):
        fork.append_token(mgr)
    print(f"After 20 fork tokens: blocks={len(fork.logical_blocks)} "
          f"util={mgr.utilization():.1%}")
