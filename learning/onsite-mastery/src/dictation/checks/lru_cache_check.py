"""lru_cache_spec 的纯断言检验：不 import solutions/。用手工构造的操作序列对拍淘汰顺序。"""
from __future__ import annotations


def check(target) -> None:
    cache_cls = target  # target 就是 LRUCache 这个类本身

    # ---- 1) 经典 LeetCode 146 示例序列，逐步验证淘汰顺序 ----
    c = cache_cls(2)
    c.put(1, 1)              # cache: {1=1}                order(MRU->LRU): [1]
    c.put(2, 2)              # cache: {1=1, 2=2}           order: [2,1]
    assert c.get(1) == 1     # 命中，1 变成 MRU              order: [1,2]
    c.put(3, 3)              # capacity=2 已满，淘汰 LRU=2   cache: {1=1,3=3} order: [3,1]
    assert c.get(2) == -1, "2 应该已经被淘汰"
    c.put(4, 4)              # 淘汰 LRU=1                    cache: {3=3,4=4} order: [4,3]
    assert c.get(1) == -1, "1 应该已经被淘汰"
    assert c.get(3) == 3
    assert c.get(4) == 4

    # ---- 2) capacity=1 边界情况 ----
    c1 = cache_cls(1)
    c1.put(10, 100)
    assert c1.get(10) == 100
    c1.put(20, 200)           # capacity=1，插入新 key 必须淘汰唯一的旧 key
    assert c1.get(10) == -1, "capacity=1 时插入新 key 应该淘汰唯一的旧 key"
    assert c1.get(20) == 200
    c1.put(20, 999)           # 更新已存在的 key，不应该有任何淘汰发生
    assert c1.get(20) == 999

    # ---- 3) put 更新已存在的 key 必须把它移动到 MRU 端，否则淘汰顺序会错 ----
    c2 = cache_cls(2)
    c2.put(1, 1)              # order: [1]
    c2.put(2, 2)              # order: [2,1]
    c2.put(1, 111)            # 更新已存在 key 1 的 value，1 应该被移到 MRU: order: [1,2]
    c2.put(3, 3)              # 容量已满，此时 LRU 应该是 2（不是 1），淘汰 2
    assert c2.get(2) == -1, "更新已存在 key 后应该刷新它的最近使用位置，2 才是真正的 LRU"
    assert c2.get(1) == 111
    assert c2.get(3) == 3

    # ---- 4) get 访问也算"最近使用"，会影响之后的淘汰顺序 ----
    c3 = cache_cls(2)
    c3.put("a", 1)            # order: [a]
    c3.put("b", 2)            # order: [b,a]
    assert c3.get("a") == 1   # 读一次 a，a 变成 MRU: order: [a,b]
    c3.put("c", 3)            # 容量已满，LRU 现在应该是 b（不是 a），淘汰 b
    assert c3.get("b") == -1, "get 访问也应该更新最近使用顺序，b 才是真正的 LRU"
    assert c3.get("a") == 1
    assert c3.get("c") == 3
