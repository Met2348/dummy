"""lru_cache_spec 的参考实现。哈希表 + 手写双向链表（带 head/tail 哨兵节点）。"""
from __future__ import annotations

from typing import Any


class _Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: Any = None, value: Any = None) -> None:
        self.key = key
        self.value = value
        self.prev: "_Node | None" = None
        self.next: "_Node | None" = None


class LRUCache:
    def __init__(self, capacity: int) -> None:
        assert capacity >= 1, "capacity 必须 >= 1"
        self.capacity = capacity
        self._map: dict[Any, _Node] = {}
        # 哨兵节点：head 之后紧跟 MRU（最近使用），tail 之前紧跟 LRU（最久未使用）
        self._head = _Node()
        self._tail = _Node()
        self._head.next = self._tail
        self._tail.prev = self._head

    def _remove(self, node: _Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_front(self, node: _Node) -> None:
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def get(self, key: Any) -> Any:
        if key not in self._map:
            return -1
        node = self._map[key]
        self._remove(node)
        self._add_front(node)
        return node.value

    def put(self, key: Any, value: Any) -> None:
        if key in self._map:
            node = self._map[key]
            node.value = value
            self._remove(node)
            self._add_front(node)
            return

        if len(self._map) >= self.capacity:
            lru_node = self._tail.prev
            self._remove(lru_node)
            del self._map[lru_node.key]

        node = _Node(key, value)
        self._map[key] = node
        self._add_front(node)
