"""一致性哈希环（带虚拟节点），纯 stdlib（hashlib + bisect）实现。

配合 backend_qa/qbank_distributed_systems.py 里 be-dsys-14（一致性哈希解决了
什么问题）用能跑的代码验证背下来的结论：
1. 环不变时，同一个 key 始终路由到同一节点。
2. 增加一个节点后，只有明显小于全部 key 数量一半的部分需要重新分布，且这部分
   迁移只流向新节点，不会打乱其它节点之间原有的分布。
"""
from __future__ import annotations

import bisect
import hashlib


class ConsistentHashRing:
    """带虚拟节点的一致性哈希环。"""

    def __init__(self, nodes: list[str] | None = None, vnodes: int = 150) -> None:
        self.vnodes = vnodes
        self._ring: dict[int, str] = {}          # hash -> 物理节点
        self._sorted_hashes: list[int] = []       # 环上所有虚拟节点哈希值，有序
        self._nodes: set[str] = set()
        for node in nodes or []:
            self.add_node(node)

    @staticmethod
    def _hash(key: str) -> int:
        return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)

    def add_node(self, node: str) -> None:
        if node in self._nodes:
            return
        self._nodes.add(node)
        for i in range(self.vnodes):
            h = self._hash(f"{node}#vn{i}")
            self._ring[h] = node
            bisect.insort(self._sorted_hashes, h)

    def remove_node(self, node: str) -> None:
        if node not in self._nodes:
            return
        self._nodes.discard(node)
        for i in range(self.vnodes):
            h = self._hash(f"{node}#vn{i}")
            del self._ring[h]
            idx = bisect.bisect_left(self._sorted_hashes, h)
            del self._sorted_hashes[idx]

    def get_node(self, key: str) -> str:
        """顺时针找环上第一个虚拟节点，返回它对应的物理节点。"""
        if not self._sorted_hashes:
            raise RuntimeError("环上没有节点")
        h = self._hash(key)
        idx = bisect.bisect(self._sorted_hashes, h)
        if idx == len(self._sorted_hashes):
            idx = 0
        return self._ring[self._sorted_hashes[idx]]


def _self_test() -> None:
    # 固定、非随机的测试 key 集合，保证结果可复现。
    keys = [f"key-{i}" for i in range(2000)]

    ring = ConsistentHashRing(["node-a", "node-b", "node-c", "node-d"], vnodes=150)

    # 1. 环不变的情况下，同一个 key 始终路由到同一节点。
    mapping_1 = {k: ring.get_node(k) for k in keys}
    mapping_2 = {k: ring.get_node(k) for k in keys}
    assert mapping_1 == mapping_2, "环不变时同一 key 的路由结果应保持一致"

    # 2. 增加一个节点后，只有明显小于全部 key 数量一半的部分需要重新分布。
    ring.add_node("node-e")
    mapping_3 = {k: ring.get_node(k) for k in keys}
    moved = sum(1 for k in keys if mapping_1[k] != mapping_3[k])
    ratio = moved / len(keys)
    assert ratio > 0.0, "增加节点后应有部分 key 迁移到新节点"
    assert ratio < 0.5, f"重新分布比例过高: {ratio:.2%}（一致性哈希应远低于 50%）"

    # 3. 一致性哈希的关键性质：迁移的 key 只会流向新节点，不会打乱其它节点间的分布。
    assert all(mapping_3[k] == "node-e" for k in keys if mapping_1[k] != mapping_3[k]), \
        "迁移的 key 应该只流向新加入的节点"

    print(
        f"[PASS] consistent_hashing: {len(keys)}个key + 稳定路由 "
        f"+ 加节点重分布{ratio:.1%}(<50%) + 迁移仅流向新节点"
    )


if __name__ == "__main__":
    _self_test()
