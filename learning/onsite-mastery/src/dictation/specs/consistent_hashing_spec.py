"""一致性哈希环（Consistent Hashing + 虚拟节点），闭卷从零手写。

面试高频度 ****。系统设计/后端 infra 岗白板代码常客，经常连着问"为什么不用普通的
hash(key) % N"、"虚拟节点解决了什么问题"。纯 stdlib（只用 hashlib），不许用任何
第三方一致性哈希库。

接口约定
--------
    class ConsistentHashRing:
        def __init__(self, virtual_nodes: int = 150): ...
        def add_node(self, node: str) -> None: ...
        def remove_node(self, node: str) -> None: ...
        def get_node(self, key: str) -> str | None: ...   # 环上没有任何节点时返回 None

    virtual_nodes : 每个物理节点在环上放多少个"虚拟节点"（副本）。虚拟节点越多，
                    key 在物理节点间分布越均匀（方差越小），但环的数据结构越大、
                    add/remove 时要处理的条目也越多——这是一个空间换均匀度的权衡。

核心思想
--------
把节点和 key 都哈希到同一个值空间（通常想象成一个环），每个 key 顺时针找到环上
第一个 >= 自己哈希值的节点，就归属那个节点。**只用一个物理节点占一个环上的点**
会导致分布严重不均匀（少数节点可能挨在一起，分走的 key 差异很大）——这就是为什么
要引入**虚拟节点**：每个物理节点在环上放 virtual_nodes 个不同哈希值的"replica"，
均匀度显著改善。

为什么不用 hash(key) % N（面试必答）
------------------------------------
普通取模在 N 变化（加/减节点）时，几乎所有 key 的归属都会变（N 变了，模的结果
基本全变），意味着几乎所有缓存/分片数据都要重新迁移——一致性哈希的核心卖点就是
**加/减一个节点时，只有大约 1/N 比例的 key 需要重新分配**，其余 key 的归属完全
不受影响。

常见实现陷阱
------------
1. **环查找的 wraparound 没处理**：用 bisect 在有序哈希值列表里找"第一个 >= key
   哈希值的位置"，如果这个位置超出了列表末尾，必须**绕回到列表开头**（环是首尾
   相连的），忘记这一步会导致哈希值比所有节点都大的 key 找不到节点。
2. **remove_node 没有清干净**：必须把该节点的**全部虚拟节点**都从环上摘掉，
   如果只删了一部分（比如虚拟节点的哈希键忘了包含副本序号，导致同名覆盖/遗漏），
   会导致这个节点"删了但没删干净"，get_node 偶尔还会返回已删除的节点。
3. **哈希函数不是节点名的纯函数**：虚拟节点的哈希值必须只依赖 `(节点名, 副本序号)`
   这个确定性输入，不能掺入插入顺序/时间戳等状态——否则加节点的顺序会影响最终
   环的形态，破坏"删除节点等价于从没加过这个节点"这条一致性哈希的核心不变量
   （check 就是照这条不变量来验证 add/remove 是否正确的）。
4. **忘了处理环为空的情况**：一个节点都没加过时 get_node 应该返回 None
   （或者按你自己的约定抛异常，但要在文档里说清楚），不能直接 IndexError 崩掉。
"""
from __future__ import annotations


class ConsistentHashRing:
    """一致性哈希环，虚拟节点支持。"""

    def __init__(self, virtual_nodes: int = 150) -> None:
        raise NotImplementedError("闭卷手写：删除这行 raise，初始化环的内部数据结构")

    def add_node(self, node: str) -> None:
        """把 node 的 virtual_nodes 个虚拟副本加入环。"""
        raise NotImplementedError("闭卷手写：删除这行 raise，实现 add_node")

    def remove_node(self, node: str) -> None:
        """把 node 的全部虚拟副本从环上摘除。"""
        raise NotImplementedError("闭卷手写：删除这行 raise，实现 remove_node")

    def get_node(self, key: str) -> str | None:
        """返回 key 应该归属的物理节点名；环为空时返回 None。"""
        raise NotImplementedError("闭卷手写：删除这行 raise，实现 get_node")
