"""LRU / LFU 缓存代码验证模块：把"我以为我懂了淘汰逻辑"变成"我跑出来验证过了"。

LRUCache 用双向链表 + 哈希表手写实现（不借助 collections.OrderedDict），
get/put 均 O(1)：链表头部是最近使用(MRU)，链表尾部是最久未使用(LRU)。

LFUCache 用哈希表 + "频次 -> 该频次下的 key 集合(按访问先后有序)"的频次桶实现，
get/put 均 O(1) 摊还：优先淘汰频次最低的 key，频次相同时淘汰其中最久未使用的那个。
频次桶内部用 collections.OrderedDict 维持"同频次内谁最久未用"的顺序，这是 stdlib，
不是重新发明哈希表本身，符合"手写核心淘汰逻辑"的要求。
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Any


class _DLLNode:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: Any = None, value: Any = None) -> None:
        self.key = key
        self.value = value
        self.prev: "_DLLNode | None" = None
        self.next: "_DLLNode | None" = None


class LRUCache:
    """双向链表 + 哈希表实现的 LRU 缓存。"""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._map: dict[Any, _DLLNode] = {}
        # 哨兵头尾节点，避免处理链表头尾的特殊 None 判断
        self._head = _DLLNode()
        self._tail = _DLLNode()
        self._head.next = self._tail
        self._tail.prev = self._head

    def _remove(self, node: _DLLNode) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def _insert_front(self, node: _DLLNode) -> None:
        """插到 head 之后，代表"最近使用"。"""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def get(self, key: Any) -> Any:
        node = self._map.get(key)
        if node is None:
            return -1
        self._remove(node)
        self._insert_front(node)
        return node.value

    def put(self, key: Any, value: Any) -> None:
        if self.capacity <= 0:
            return
        node = self._map.get(key)
        if node is not None:
            node.value = value
            self._remove(node)
            self._insert_front(node)
            return
        if len(self._map) >= self.capacity:
            lru_node = self._tail.prev  # 链表尾部前一个即最久未使用
            self._remove(lru_node)
            del self._map[lru_node.key]
        new_node = _DLLNode(key, value)
        self._map[key] = new_node
        self._insert_front(new_node)


class LFUCache:
    """哈希表 + 频次桶实现的 LFU 缓存：淘汰频次最低者，频次相同淘汰最久未使用者。"""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.min_freq = 0
        self._key_to_val: dict[Any, Any] = {}
        self._key_to_freq: dict[Any, int] = {}
        # freq -> OrderedDict(key -> None)，插入顺序即"最近使用顺序"，
        # popitem(last=False) 弹出的就是该频次下最久未使用的 key
        self._freq_to_keys: dict[int, OrderedDict] = defaultdict(OrderedDict)

    def _touch(self, key: Any) -> None:
        """访问命中后，把 key 从旧频次桶挪到新频次桶（频次+1）。"""
        freq = self._key_to_freq[key]
        del self._freq_to_keys[freq][key]
        if not self._freq_to_keys[freq] and self.min_freq == freq:
            self.min_freq += 1
        new_freq = freq + 1
        self._key_to_freq[key] = new_freq
        self._freq_to_keys[new_freq][key] = None

    def get(self, key: Any) -> Any:
        if key not in self._key_to_val:
            return -1
        self._touch(key)
        return self._key_to_val[key]

    def put(self, key: Any, value: Any) -> None:
        if self.capacity <= 0:
            return
        if key in self._key_to_val:
            self._key_to_val[key] = value
            self._touch(key)
            return
        if len(self._key_to_val) >= self.capacity:
            # 从 min_freq 桶里淘汰最久未使用的（OrderedDict 里最早插入的）
            oldest_key, _ = self._freq_to_keys[self.min_freq].popitem(last=False)
            del self._key_to_val[oldest_key]
            del self._key_to_freq[oldest_key]
        self._key_to_val[key] = value
        self._key_to_freq[key] = 1
        self._freq_to_keys[1][key] = None
        self.min_freq = 1


def _self_test() -> None:
    # ---- LRU：容量2，验证"更新已存在的key不淘汰、且刷新为最近使用" ----
    lru = LRUCache(2)
    lru.put(1, 100)
    lru.put(2, 200)
    lru.put(1, 111)            # 更新已存在的key：值变新，且1变为最近使用
    lru.put(3, 300)            # 容量已满，应淘汰最久未用的2（不是1）
    assert lru.get(2) == -1, "2 应该已被淘汰"
    assert lru.get(1) == 111
    assert lru.get(3) == 300

    # ---- LRU：容量3，自行设计的更长序列，验证淘汰顺序始终是"最久未使用" ----
    lru2 = LRUCache(3)
    assert lru2.get("a") == -1                # 空缓存查询未命中
    lru2.put("a", 1)
    lru2.put("b", 2)
    lru2.put("c", 3)                           # MRU->LRU: c, b, a
    assert lru2.get("a") == 1                  # a 变为 MRU -> a, c, b
    lru2.put("d", 4)                           # 容量满，淘汰 LRU 的 b -> d, a, c
    assert lru2.get("b") == -1
    assert lru2.get("c") == 3                  # c 变为 MRU -> c, d, a
    lru2.put("e", 5)                           # 容量满，淘汰 LRU 的 a -> e, c, d
    assert lru2.get("a") == -1
    assert lru2.get("d") == 4
    assert lru2.get("c") == 3
    assert lru2.get("e") == 5

    # ---- LRU：容量1，最小边界 ----
    lru3 = LRUCache(1)
    lru3.put("x", 1)
    lru3.put("y", 2)                           # 淘汰 x
    assert lru3.get("x") == -1
    assert lru3.get("y") == 2

    # ---- LFU：容量2，验证"淘汰最低频次" + "频次相同淘汰最久未使用" ----
    lfu = LFUCache(2)
    lfu.put(1, 1)
    lfu.put(2, 2)
    assert lfu.get(1) == 1                     # freq(1)=2, freq(2)=1
    lfu.put(3, 3)                              # 淘汰频次最低的2（不是1）
    assert lfu.get(2) == -1
    assert lfu.get(3) == 3                     # 此后 freq(1)=2, freq(3)=2，频次打平
    lfu.put(4, 4)                              # 频次相同时淘汰最久未使用的1（1的最近一次访问早于3）
    assert lfu.get(1) == -1, "频次相同应淘汰最久未使用的1，不是3"
    assert lfu.get(3) == 3
    assert lfu.get(4) == 4

    # ---- LFU：容量0边界，put/get都应是安全的no-op/miss ----
    lfu0 = LFUCache(0)
    lfu0.put(1, 1)
    assert lfu0.get(1) == -1

    print("[PASS] lru_lfu_cache: LRU(双向链表+哈希表) + LFU(频次桶) 淘汰顺序均验证通过")


if __name__ == "__main__":
    _self_test()
