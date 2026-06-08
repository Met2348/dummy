"""KV cache mock with PagedAttention-style block allocation."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class KVBlock:
    block_id: int
    tokens: list[int] = field(default_factory=list)
    keys: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)


class KVCache:
    def __init__(self, block_size: int = 4, max_blocks: int = 64):
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.blocks: dict[int, KVBlock] = {}
        self.next_block_id = 0
        self.free_blocks: list[int] = []
        self.seq_to_blocks: dict[str, list[int]] = {}
        self.shared_prefix: dict[str, int] = {}

    def _allocate_block(self) -> int:
        if self.free_blocks:
            bid = self.free_blocks.pop()
            self.blocks[bid] = KVBlock(block_id=bid)
            return bid
        if len(self.blocks) >= self.max_blocks:
            raise MemoryError("KV cache full")
        bid = self.next_block_id
        self.next_block_id += 1
        self.blocks[bid] = KVBlock(block_id=bid)
        return bid

    def add_seq(self, seq_id: str, tokens: list[int]) -> list[int]:
        block_ids: list[int] = []
        current_block: int | None = None
        for i, tok in enumerate(tokens):
            if i % self.block_size == 0:
                current_block = self._allocate_block()
                block_ids.append(current_block)
            self.blocks[current_block].tokens.append(tok)
            self.blocks[current_block].keys.append(float(tok) * 1.1)
            self.blocks[current_block].values.append(float(tok) * 0.9)
        self.seq_to_blocks[seq_id] = block_ids
        return block_ids

    def free_seq(self, seq_id: str) -> None:
        for bid in self.seq_to_blocks.pop(seq_id, []):
            if bid in self.blocks:
                self.free_blocks.append(bid)
                del self.blocks[bid]

    def share_prefix(self, prefix_id: str, seq_id: str, n_prefix_tokens: int) -> int:
        if prefix_id not in self.shared_prefix:
            return 0
        prefix_blocks = self.seq_to_blocks.get(prefix_id, [])
        n_blocks_to_share = min(
            len(prefix_blocks),
            (n_prefix_tokens + self.block_size - 1) // self.block_size,
        )
        if not n_blocks_to_share:
            return 0
        shared = prefix_blocks[:n_blocks_to_share]
        self.seq_to_blocks.setdefault(seq_id, [])
        # Reuse block ids instead of allocating new blocks.
        self.seq_to_blocks[seq_id] = shared + self.seq_to_blocks.get(seq_id, [])
        return len(shared)

    def mark_prefix(self, prefix_id: str) -> None:
        self.shared_prefix[prefix_id] = 1

    def usage(self) -> dict:
        return {
            "n_blocks_used": len(self.blocks),
            "n_blocks_free": len(self.free_blocks),
            "n_seqs": len(self.seq_to_blocks),
        }


def _self_test() -> None:
    kv = KVCache(block_size=4, max_blocks=20)

    blocks_a = kv.add_seq("seq_a", list(range(10)))
    assert len(blocks_a) == 3
    assert kv.usage()["n_blocks_used"] == 3

    kv.mark_prefix("seq_a")

    blocks_b = kv.add_seq("seq_b", list(range(10, 20)))
    assert kv.usage()["n_blocks_used"] == 6

    shared = kv.share_prefix("seq_a", "seq_b", n_prefix_tokens=8)
    assert shared == 2

    kv.free_seq("seq_a")
    used_after_free = kv.usage()
    assert used_after_free["n_blocks_free"] >= 1
    print("[OK] kv_cache_mock._self_test passed")


if __name__ == "__main__":
    _self_test()
