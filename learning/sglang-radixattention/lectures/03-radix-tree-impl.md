# L03 · Radix Tree 实现细节

## 1 · 数据结构
```python
@dataclass
class Node:
    token_ids: List[int]
    children: Dict[int, "Node"]    # 首 token -> 子节点
    parent: Optional["Node"]
    kv_slots: List[int]             # 对应物理 KV 位置（占位）
    refcount: int = 0
    last_access: float = 0.0

class RadixTree:
    root: Node
    n_tokens: int = 0
    cap: int = 100_000
```

## 2 · match 操作
```python
def match(self, prefix: List[int]) -> Tuple[Node, int]:
    """返回 (deepest hit node, n_matched_tokens)"""
    node = self.root
    i = 0
    while i < len(prefix):
        child = node.children.get(prefix[i])
        if child is None:
            return node, i
        # 在 child.token_ids 上 walk
        for j, t in enumerate(child.token_ids):
            if i + j >= len(prefix) or prefix[i + j] != t:
                # partial match → split
                return child, i + j  # caller 决定是否 split
        node = child
        i += len(child.token_ids)
    return node, i
```

## 3 · split 操作
当 prefix 在节点中间分叉：
```python
def split(self, node: Node, at: int) -> Node:
    """把 node 拆成 head[0:at] + tail[at:]"""
    head_tokens = node.token_ids[:at]
    tail_tokens = node.token_ids[at:]
    head = Node(token_ids=head_tokens, parent=node.parent, kv_slots=node.kv_slots[:at])
    tail = Node(token_ids=tail_tokens, parent=head, kv_slots=node.kv_slots[at:])
    # rewire
    parent = node.parent
    parent.children[head_tokens[0]] = head
    head.children = {tail_tokens[0]: tail}
    tail.children = node.children
    for c in tail.children.values():
        c.parent = tail
    return head
```

## 4 · insert 操作
```python
def insert(self, prefix: List[int], new_kv_slots: List[int]) -> Node:
    node, matched = self.match(prefix)
    if matched < ...:
        node = self.split(node, matched - parent_offset)
    new_node = Node(
        token_ids=prefix[matched:],
        parent=node,
        kv_slots=new_kv_slots,
    )
    node.children[prefix[matched]] = new_node
    return new_node
```

## 5 · evict LRU
- 维护 leaf 集合的 min-heap by `last_access`
- 满时 pop 最小 + refcount=0 的叶子
- 释放其 kv_slots 给 pool
- 它的 parent 若变成叶子，加回 heap

## 6 · refcount
- 请求开始 → 走 match → 沿路径所有节点 refcount++
- 请求结束 → refcount--
- evict 时跳过 refcount > 0 的节点

## 7 · 一致性检查（实现要点）
- token 序列从 root 到叶 == 该 request 的完整 prompt+output
- evict 后 root 不变
- refcount 与 active request 数一致

## 8 · 实现
继续在 `radix_tree.py`，附 5 个测试。
