"""consistent_hashing_spec 的参考实现。纯 stdlib：hashlib + bisect。"""
from __future__ import annotations

import bisect
import hashlib


class ConsistentHashRing:
    def __init__(self, virtual_nodes: int = 150) -> None:
        self.virtual_nodes = virtual_nodes
        self._ring: dict[int, str] = {}       # 哈希值 -> 物理节点名
        self._sorted_hashes: list[int] = []    # 有序哈希值列表，用于 bisect 查找
        self._nodes: set[str] = set()

    @staticmethod
    def _hash(s: str) -> int:
        return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)

    def add_node(self, node: str) -> None:
        if node in self._nodes:
            return
        self._nodes.add(node)
        for i in range(self.virtual_nodes):
            h = self._hash(f"{node}#{i}")
            self._ring[h] = node
            bisect.insort(self._sorted_hashes, h)

    def remove_node(self, node: str) -> None:
        if node not in self._nodes:
            return
        self._nodes.discard(node)
        for i in range(self.virtual_nodes):
            h = self._hash(f"{node}#{i}")
            if h in self._ring:
                del self._ring[h]
                idx = bisect.bisect_left(self._sorted_hashes, h)
                if idx < len(self._sorted_hashes) and self._sorted_hashes[idx] == h:
                    self._sorted_hashes.pop(idx)

    def get_node(self, key: str) -> str | None:
        if not self._sorted_hashes:
            return None
        h = self._hash(key)
        idx = bisect.bisect_left(self._sorted_hashes, h)
        if idx == len(self._sorted_hashes):
            idx = 0  # 绕回环的开头
        return self._ring[self._sorted_hashes[idx]]
