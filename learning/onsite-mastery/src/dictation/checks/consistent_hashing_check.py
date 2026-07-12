"""consistent_hashing_spec 的纯断言检验：不 import solutions/。

策略：不去猜测被测实现内部用的具体哈希函数，而是验证一致性哈希的核心不变量——
"增量 add_node 的结果必须和直接用全量节点建环完全一致"、"remove_node 之后必须
和从没加过这个节点完全一致"，这两条对任何正确实现都成立，与内部哈希函数无关，
是黑盒也能验证的强结构性质。
"""
from __future__ import annotations


def _sample_keys(n: int) -> list[str]:
    return [f"key-{i}" for i in range(n)]


def check(target) -> None:
    ring_cls = target  # target 就是 ConsistentHashRing 这个类本身
    base_nodes = ["node-A", "node-B", "node-C", "node-D"]
    keys = _sample_keys(2000)

    ring = ring_cls(virtual_nodes=200)
    for n in base_nodes:
        ring.add_node(n)

    before = {k: ring.get_node(k) for k in keys}
    assert set(before.values()) <= set(base_nodes), "get_node 返回了没加过的节点"
    assert all(v is not None for v in before.values())

    # ---- add_node 检验 ----
    ring.add_node("node-E")
    after_add = {k: ring.get_node(k) for k in keys}

    # 1) 增量加一个节点，结果必须和"直接用全部 5 个节点建环"完全一致（顺序无关性）
    fresh_5 = ring_cls(virtual_nodes=200)
    for n in base_nodes + ["node-E"]:
        fresh_5.add_node(n)
    fresh_after_add = {k: fresh_5.get_node(k) for k in keys}
    assert after_add == fresh_after_add, (
        "增量 add_node 的结果必须和直接用全量 5 个节点建环完全一致"
        "（如果哈希值依赖了插入顺序/历史状态就会在这里出错）"
    )

    # 2) 只有大约 1/(N+1) 比例的 key 需要重新分配，且这些 key 必须全部迁移到新节点
    moved = [k for k in keys if before[k] != after_add[k]]
    assert len(moved) > 0, "加一个新节点之后应该有部分 key 发生迁移"
    assert all(after_add[k] == "node-E" for k in moved), (
        "迁移的 key 必须全部落到新加的节点上，不能有 key 被甩到其它旧节点"
    )
    frac = len(moved) / len(keys)
    assert 0.08 < frac < 0.35, (
        f"迁移比例应接近理论值 1/5=0.20（节点数从 4 变 5），实得 {frac:.3f}"
    )

    # ---- remove_node 检验 ----
    ring.remove_node("node-B")
    after_remove = {k: ring.get_node(k) for k in keys}
    assert "node-B" not in set(after_remove.values()), "remove_node 之后不应再有 key 归属被删节点"

    # 3) 删除节点之后，必须和"从没加过这个节点"建的环完全一致
    #    （原本落在被删节点上的 key，要正确迁移到环上顺时针方向的下一个节点）
    fresh_4 = ring_cls(virtual_nodes=200)
    for n in ["node-A", "node-C", "node-D", "node-E"]:
        fresh_4.add_node(n)
    fresh_after_remove = {k: fresh_4.get_node(k) for k in keys}
    assert after_remove == fresh_after_remove, (
        "remove_node 之后的结果必须和从没加过该节点建的环完全一致"
        "（虚拟节点有没有摘干净、环上位置查找的 wraparound 有没有处理对，都会在这里暴露）"
    )

    # ---- 环为空的边界情况 ----
    empty_ring = ring_cls(virtual_nodes=50)
    assert empty_ring.get_node("anything") is None, "环为空时 get_node 应返回 None"
