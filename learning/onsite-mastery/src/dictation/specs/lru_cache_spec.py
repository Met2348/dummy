"""LRU Cache（O(1) get/put），闭卷从零手写。

面试高频度 *****。系统设计+数据结构的经典交叉题（LeetCode 146），几乎每个后端/
infra 岗位都会问。**禁止偷懒**：不能直接用 `collections.OrderedDict`
（它内部已经帮你实现好了双向链表+哈希表，用它相当于没写这道题），也不能用
`functools.lru_cache`（那是装饰器，不是这道题要考的数据结构）。要求自己维护
双向链表节点。

接口约定
--------
    class LRUCache:
        def __init__(self, capacity: int): ...
        def get(self, key) -> Any: ...       # 命中返回 value；未命中返回 -1
        def put(self, key, value) -> None: ...

    capacity : 缓存最多能放多少个 key，capacity >= 1（capacity=1 是必考边界情况）

    约定：`get` 未命中时返回 **-1**（沿用 LeetCode 146 的经典约定，隐含假设
    value 本身不会是 -1；如果面试官要求更严谨的语义，可以改成抛 KeyError 或者
    返回 None，但要在这里写清楚你选的是哪种约定）。

正确性要求（O(1) 摊还时间复杂度，这是这道题的重点，不是"能跑对就行"）
--------------------------------------------------------------------
- `get(key)`：命中要把该 key 标记为"最近使用"（moved to MRU 端），未命中返回 -1。
- `put(key, value)`：
    - key 已存在：更新 value，并把它标记为"最近使用"。
    - key 不存在且已达 capacity：先淘汰"最久未使用"（LRU 端）的 key，再插入新 key。
    - key 不存在且未达 capacity：直接插入。
- 全程只能用 O(1) 操作（哈希表查找 + 双向链表节点的 O(1) 摘除/插入），
  不允许用线性扫描去找"最久未使用的是谁"。

面试常问
--------
- 为什么要用**双向**链表，单向链表不行吗？—— 删除一个节点需要知道它的 prev
  才能把 prev.next 接到 next 上；单向链表要找 prev 得从头遍历，退化成 O(n)。
- 为什么要用两个哨兵节点（sentinel head/tail），而不是让 head/tail 直接指向
  真实节点？—— 哨兵节点让"链表为空"、"只有一个节点"这些边界情况不需要特判
  （永远有 head.next 和 tail.prev 可以安全访问），这是本题减少 corner-case bug
  的关键技巧。

常见实现陷阱
------------
1. **只在 put 时更新访问顺序，get 时忘了更新**：LRU 的"最近使用"包括**读**，
   不只是写；get 命中之后必须把节点挪到 MRU 端，否则淘汰顺序会错。
2. **put 更新已存在 key 时忘了把它挪到 MRU 端**：只更新了 value，链表位置
   没变，之后淘汰顺序会错（这个 key 明明刚被访问过，却可能被当成最久未用的
   淘汰掉）。
3. **淘汰时只从链表摘除，忘了同步从哈希表里删除**：留下的哈希表脏条目会让
   `key in cache` 类的判断出错，也会让 capacity 计数对不上。
4. **capacity=1 的边界**：这时候链表里最多只有一个"真实"节点（两个哨兵之间），
   插入第二个 key 必须正确淘汰第一个，很多手写实现在这个边界上会有指针错误。
"""
from __future__ import annotations

from typing import Any


class LRUCache:
    """O(1) get/put 的 LRU 缓存：哈希表 + 手写双向链表，禁止用 OrderedDict/lru_cache。"""

    def __init__(self, capacity: int) -> None:
        raise NotImplementedError("闭卷手写：删除这行 raise，初始化容量、哈希表、双向链表哨兵节点")

    def get(self, key: Any) -> Any:
        """命中返回 value 并把该 key 标记为最近使用；未命中返回 -1。"""
        raise NotImplementedError("闭卷手写：删除这行 raise，实现 get")

    def put(self, key: Any, value: Any) -> None:
        """写入/更新 key，必要时淘汰最久未使用的 key。"""
        raise NotImplementedError("闭卷手写：删除这行 raise，实现 put")
