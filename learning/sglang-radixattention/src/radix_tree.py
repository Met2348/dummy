"""Educational radix tree backing SGLang's RadixAttention.

Nodes hold a *segment* of token ids (radix-compressed); when a new request
shares a partial prefix we split the segment. KV is modelled as integer slot
ids; a real engine would map these to PagedKvPool blocks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time


@dataclass
class Node:
    token_ids: List[int] = field(default_factory=list)
    children: Dict[int, "Node"] = field(default_factory=dict)
    parent: Optional["Node"] = None
    kv_slots: List[int] = field(default_factory=list)
    refcount: int = 0
    last_access: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return not self.children


@dataclass
class RadixTree:
    cap: int = 100_000          # max tokens stored
    n_tokens: int = 0
    next_slot: int = 0
    hits: int = 0
    misses: int = 0

    def __post_init__(self) -> None:
        self.root = Node()

    # ---- match -------------------------------------------------------------

    def match(self, prefix: List[int]) -> Tuple[Node, int]:
        """Walk as far as `prefix` matches the tree.

        Returns (deepest matched node, n_tokens_matched).  When the match ends
        midway through a node's `token_ids`, the returned node is that node
        and caller is responsible for splitting before insertion.
        """
        node = self.root
        i = 0
        while i < len(prefix):
            child = node.children.get(prefix[i])
            if child is None:
                return node, i
            # walk through child's segment
            seg = child.token_ids
            j = 0
            while j < len(seg) and i + j < len(prefix) and seg[j] == prefix[i + j]:
                j += 1
            if j < len(seg):
                # Partial match through this child: split here.
                node = child
                i += j
                return node, i
            # Full child consumed: continue from grandchildren.
            node = child
            i += j
        return node, i

    # ---- split -------------------------------------------------------------

    def _split(self, node: Node, at: int) -> Node:
        """Split `node.token_ids` at offset `at`; returns the head.

        `node` becomes the *tail*: its parent and head are updated, children
        remain on the tail (since the old leaf state lives there).
        """
        head_tokens = node.token_ids[:at]
        tail_tokens = node.token_ids[at:]
        head_slots = node.kv_slots[:at]
        tail_slots = node.kv_slots[at:]
        head = Node(
            token_ids=head_tokens,
            parent=node.parent,
            kv_slots=head_slots,
            last_access=node.last_access,
            refcount=node.refcount,
        )
        # Rewire parent to head.
        node.parent.children[head_tokens[0]] = head
        # mutate node into tail
        node.token_ids = tail_tokens
        node.kv_slots = tail_slots
        node.parent = head
        head.children = {tail_tokens[0]: node}
        return head

    # ---- insert ------------------------------------------------------------

    def insert(self, prefix: List[int]) -> Tuple[Node, int]:
        """Insert/lookup prefix, return (leaf_node, n_tokens_matched_existing)."""
        node, matched = self.match(prefix)
        # may need to split if match landed inside a node
        if node is not self.root and matched > 0:
            # how far into `node`'s segment did we land?
            # We can compute by walking from root summing each ancestor's len.
            depth = self._depth(node)
            into = matched - (depth - len(node.token_ids))
            if 0 < into < len(node.token_ids):
                node = self._split(node, into)
        # add new node for remainder
        remainder = prefix[matched:]
        if remainder:
            new_slots = list(range(self.next_slot, self.next_slot + len(remainder)))
            self.next_slot += len(remainder)
            new_node = Node(
                token_ids=remainder,
                parent=node,
                kv_slots=new_slots,
                last_access=time.perf_counter(),
            )
            node.children[remainder[0]] = new_node
            self.n_tokens += len(remainder)
            self.misses += len(remainder)
            node = new_node
        self.hits += matched
        node.last_access = time.perf_counter()
        return node, matched

    def _depth(self, node: Node) -> int:
        d = 0
        cur = node
        while cur is not self.root:
            d += len(cur.token_ids)
            cur = cur.parent
        return d

    # ---- refcount ----------------------------------------------------------

    def acquire(self, leaf: Node) -> None:
        cur = leaf
        while cur is not self.root:
            cur.refcount += 1
            cur = cur.parent

    def release(self, leaf: Node) -> None:
        cur = leaf
        while cur is not self.root:
            cur.refcount = max(0, cur.refcount - 1)
            cur = cur.parent

    # ---- evict -------------------------------------------------------------

    def _leaves_zero_ref(self) -> List[Node]:
        leaves = []
        stack = [self.root]
        while stack:
            n = stack.pop()
            if n is not self.root and n.is_leaf and n.refcount == 0:
                leaves.append(n)
            stack.extend(n.children.values())
        return leaves

    def evict(self, want_tokens: int) -> int:
        freed = 0
        while freed < want_tokens:
            leaves = self._leaves_zero_ref()
            if not leaves:
                break
            victim = min(leaves, key=lambda n: n.last_access)
            freed += len(victim.token_ids)
            self.n_tokens -= len(victim.token_ids)
            del victim.parent.children[victim.token_ids[0]]
        return freed

    # ---- stats -------------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / max(total, 1)

    def total_nodes(self) -> int:
        count = 0
        stack = [self.root]
        while stack:
            n = stack.pop()
            if n is not self.root:
                count += 1
            stack.extend(n.children.values())
        return count


if __name__ == "__main__":
    tree = RadixTree()
    prompts = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 8, 9],
        [10, 11, 12],
    ]
    for p in prompts:
        leaf, matched = tree.insert(p)
        print(f"insert {p}: matched={matched} n_tokens={tree.n_tokens} nodes={tree.total_nodes()}")
    print(f"hit_rate={tree.hit_rate:.3f}")
