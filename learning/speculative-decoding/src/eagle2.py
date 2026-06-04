"""EAGLE-2 — dynamic draft tree with confidence-based pruning."""
from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Callable, List, Tuple
import math
import random

from common import softmax, sample_from


@dataclass
class TreeNode:
    tokens: List[int] = field(default_factory=list)
    logprob: float = 0.0
    depth: int = 0


def topk(probs: List[float], k: int) -> List[int]:
    return sorted(range(len(probs)), key=lambda i: -probs[i])[:k]


def build_dynamic_tree(
    target_fn: Callable[[List[int]], List[float]],
    prefix: List[int],
    K: int = 8,
    max_depth: int = 5,
    branch: int = 3,
    noise: float = 0.25,
    rng: random.Random = None,
) -> List[TreeNode]:
    """Beam-search expand a draft tree; keep K most promising leaf paths."""
    rng = rng or random.Random(0)
    leaves: List[Tuple[float, int, TreeNode]] = []     # (-lp, id, node)
    counter = 0
    root = TreeNode()
    heappush(leaves, (0.0, counter, root))
    finals: List[TreeNode] = []
    while leaves and len(finals) < K:
        neg_lp, _, leaf = heappop(leaves)
        if leaf.depth >= max_depth:
            finals.append(leaf)
            continue
        local = prefix + leaf.tokens
        true_p = target_fn(local)
        # noisy "draft distribution" to mimic EAGLE-2 draft model
        noisy = [max(p + rng.uniform(-noise, noise), 0.0) for p in true_p]
        s = sum(noisy)
        q = [p / s for p in noisy]
        for tid in topk(q, branch):
            new_lp = -neg_lp + math.log(max(q[tid], 1e-12))
            child = TreeNode(tokens=leaf.tokens + [tid], logprob=new_lp, depth=leaf.depth + 1)
            counter += 1
            heappush(leaves, (-new_lp, counter, child))
    while leaves and len(finals) < K:
        _, _, node = heappop(leaves)
        finals.append(node)
    return finals
