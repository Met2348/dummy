"""B+ 树教学模拟（简化版，非工业级实现）。

说明：这是**简化教学版本**，固定阶数（每个节点最多 MAX_KEYS 个 key，超过则分裂），
只实现 insert / search / range_query，不实现删除、合并、再平衡等工业级细节。
目的是验证 B+ 树的两个关键性质：

1. **范围查询效率**：叶子节点用单向链表串联（每个叶子有 .next 指针），
   range_query() 只需要先用类似 B 树的方式定位到起始叶子，然后沿着叶子链表
   顺序向后扫描，不需要每次都从根节点重新遍历整棵树。
2. **查找/插入结果正确**：insert 按 key 有序落到叶子，search 能正确找到已插入的
   key 对应的 value，中间节点只存路由用的分隔 key（不存 value）。

`_self_test()` 用暴力法（把所有插入过的 key 放进一个 Python 有序 list，
线性扫描做范围查询）对拍本文件的 B+ 树实现，覆盖 30 个不同的 key。
"""
from __future__ import annotations

MAX_KEYS = 3  # 阶数 4 的简化 B+ 树：每个节点最多 3 个 key，插入后超过则分裂


class _LeafNode:
    __slots__ = ("keys", "values", "next")

    def __init__(self) -> None:
        self.keys: list[int] = []
        self.values: list[object] = []
        self.next: "_LeafNode | None" = None       # 叶子链表：范围查询靠它顺序遍历


class _InternalNode:
    __slots__ = ("keys", "children")

    def __init__(self) -> None:
        self.keys: list[int] = []                  # 分隔 key，纯路由用，不存 value
        self.children: list["_LeafNode | _InternalNode"] = []


class BPlusTree:
    def __init__(self) -> None:
        self.root: "_LeafNode | _InternalNode" = _LeafNode()

    # -- 查找 -----------------------------------------------------------
    def _find_leaf(self, key: int) -> _LeafNode:
        node = self.root
        while isinstance(node, _InternalNode):
            i = 0
            # children[i] 存放 key < node.keys[i] 的部分；key >= 分隔 key 就往右走
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            node = node.children[i]
        return node

    def search(self, key: int) -> object | None:
        leaf = self._find_leaf(key)
        for k, v in zip(leaf.keys, leaf.values):
            if k == key:
                return v
        return None

    # -- 插入 -----------------------------------------------------------
    def insert(self, key: int, value: object) -> None:
        result = self._insert(self.root, key, value)
        if result is not None:
            sep_key, right = result
            new_root = _InternalNode()
            new_root.keys = [sep_key]
            new_root.children = [self.root, right]
            self.root = new_root

    def _insert(self, node, key: int, value: object):
        if isinstance(node, _LeafNode):
            i = 0
            while i < len(node.keys) and node.keys[i] < key:
                i += 1
            if i < len(node.keys) and node.keys[i] == key:
                node.values[i] = value          # key 已存在：更新 value
                return None
            node.keys.insert(i, key)
            node.values.insert(i, value)
            if len(node.keys) <= MAX_KEYS:
                return None
            return self._split_leaf(node)

        i = 0
        while i < len(node.keys) and key >= node.keys[i]:
            i += 1
        result = self._insert(node.children[i], key, value)
        if result is None:
            return None
        sep_key, right = result
        node.keys.insert(i, sep_key)
        node.children.insert(i + 1, right)
        if len(node.keys) <= MAX_KEYS:
            return None
        return self._split_internal(node)

    @staticmethod
    def _split_leaf(node: _LeafNode):
        mid = len(node.keys) // 2
        right = _LeafNode()
        right.keys = node.keys[mid:]
        right.values = node.values[mid:]
        node.keys = node.keys[:mid]
        node.values = node.values[:mid]
        right.next = node.next
        node.next = right                        # 维护叶子链表
        return right.keys[0], right              # 分隔 key = 右半区最小 key（B+ 树叶子仍保留它）

    @staticmethod
    def _split_internal(node: _InternalNode):
        mid = len(node.keys) // 2
        sep_key = node.keys[mid]                 # 内部节点分裂：中间 key 上移，本节点不再保留
        right = _InternalNode()
        right.keys = node.keys[mid + 1:]
        right.children = node.children[mid + 1:]
        node.keys = node.keys[:mid]
        node.children = node.children[:mid + 1]
        return sep_key, right

    # -- 范围查询：定位起始叶子后，沿叶子链表顺序扫描，不回到根 --------------
    def range_query(self, lo: int, hi: int) -> list[tuple[int, object]]:
        node: "_LeafNode | None" = self._find_leaf(lo)
        out: list[tuple[int, object]] = []
        while node is not None:
            for k, v in zip(node.keys, node.values):
                if k > hi:
                    return out
                if k >= lo:
                    out.append((k, v))
            node = node.next
        return out

    # -- 辅助：从最左叶子开始顺序遍历全部 key，验证叶子链表本身是有序、连通的 --
    def _leftmost_leaf(self) -> _LeafNode:
        node = self.root
        while isinstance(node, _InternalNode):
            node = node.children[0]
        return node

    def all_keys_via_leaf_chain(self) -> list[int]:
        out: list[int] = []
        node: "_LeafNode | None" = self._leftmost_leaf()
        while node is not None:
            out.extend(node.keys)
            node = node.next
        return out


def _self_test() -> None:
    # 30 个不同的 key，插入顺序刻意打乱（非升序/非随机模块，纯手写固定顺序）
    keys = [
        50, 20, 70, 10, 30, 60, 80, 5, 15, 25,
        35, 55, 65, 75, 85, 1, 8, 12, 18, 22,
        28, 32, 38, 52, 58, 62, 68, 72, 78, 92,
    ]
    assert len(keys) == len(set(keys)) == 30

    tree = BPlusTree()
    brute: list[int] = []                        # 暴力对拍：有序 list + 线性扫描
    for k in keys:
        tree.insert(k, k * 10)
        pos = 0
        while pos < len(brute) and brute[pos] < k:
            pos += 1
        brute.insert(pos, k)

    # 1) 查找正确性：每个插入过的 key 都应该能查到正确的 value
    for k in keys:
        assert tree.search(k) == k * 10, f"search({k}) 错误"

    # 2) 查找不存在的 key 应返回 None
    for missing in (-1, 0, 999, 1000):
        assert tree.search(missing) is None

    # 3) 更新已存在 key 的 value
    tree.insert(50, "updated")
    assert tree.search(50) == "updated"
    tree.insert(50, 500)  # 改回来，避免影响后面的范围查询断言

    # 4) 叶子链表本身应该保持全局有序、且包含全部 key（验证性质 1 的基础：链表连通有序）
    assert tree.all_keys_via_leaf_chain() == sorted(brute)

    # 5) 范围查询与暴力线性扫描对拍：覆盖边界、空区间、跨多个叶子的区间
    test_ranges = [
        (1, 100), (0, 10), (20, 60), (55, 55), (86, 91),
        (-5, 5), (90, 200), (33, 33), (10, 85), (999, 1000),
    ]
    for lo, hi in test_ranges:
        expected = [(k, k * 10 if k != 50 else 500) for k in sorted(brute) if lo <= k <= hi]
        got = tree.range_query(lo, hi)
        assert got == expected, f"range_query({lo},{hi}) 期望 {expected} 实际 {got}"

    print(
        f"[PASS] bplus_tree_sim: {len(keys)}个key插入/查找/更新对拍一致 "
        f"+ 叶子链表顺序遍历一致 + {len(test_ranges)}组range_query与暴力法对拍一致"
    )


if __name__ == "__main__":
    _self_test()
