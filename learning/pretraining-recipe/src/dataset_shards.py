"""Memmap shard dataset (nanoGPT 风格)."""
from __future__ import annotations

import os
import numpy as np
import torch
from pathlib import Path


def write_shard(token_ids: list, path: str, dtype=np.uint16) -> None:
    arr = np.array(token_ids, dtype=dtype)
    arr.tofile(path)


def load_shard(path: str, dtype=np.uint16):
    return np.memmap(path, dtype=dtype, mode="r")


def sample_batch(data, seq_len: int, batch_size: int, rng):
    """随机起点采 batch."""
    n = len(data) - seq_len - 1
    starts = rng.integers(0, n, size=batch_size)
    x = np.stack([np.asarray(data[i:i + seq_len]) for i in starts])
    y = np.stack([np.asarray(data[i + 1:i + seq_len + 1]) for i in starts])
    return (torch.from_numpy(x.astype(np.int64)),
            torch.from_numpy(y.astype(np.int64)))


class ShardManager:
    """多 shard 顺序遍历 + resume 支持."""
    def __init__(self, shard_paths: list, dtype=np.uint16):
        self.paths = shard_paths
        self.dtype = dtype
        self.cur_shard_idx = 0
        self.cur_offset = 0
        self.shard = None
        self._open_shard()

    def _open_shard(self):
        self.shard = load_shard(self.paths[self.cur_shard_idx],
                                 self.dtype)

    def next_seq(self, seq_len: int) -> np.ndarray:
        if self.cur_offset + seq_len + 1 > len(self.shard):
            self.cur_shard_idx = (self.cur_shard_idx + 1) % len(self.paths)
            self.cur_offset = 0
            self._open_shard()
        seq = np.asarray(self.shard[self.cur_offset:
                                     self.cur_offset + seq_len + 1])
        self.cur_offset += seq_len
        return seq

    def state(self) -> dict:
        return {"shard_idx": self.cur_shard_idx, "offset": self.cur_offset}

    def restore(self, state: dict) -> None:
        self.cur_shard_idx = state["shard_idx"]
        self.cur_offset = state["offset"]
        self._open_shard()


if __name__ == "__main__":
    Path("/tmp/test_shards").mkdir(exist_ok=True, parents=True)
    for i in range(3):
        ids = list(range(i * 1000, (i + 1) * 1000))
        write_shard(ids, f"/tmp/test_shards/shard_{i}.bin")

    paths = [f"/tmp/test_shards/shard_{i}.bin" for i in range(3)]
    mgr = ShardManager(paths)

    for _ in range(5):
        seq = mgr.next_seq(100)
        print(f"  state={mgr.state()}, first={seq[0]} last={seq[-1]}")
