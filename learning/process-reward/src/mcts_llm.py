"""MCTS for LLM — Stream of Search / rStar-Math 风格简化.

Tree:
    root = question
    node = partial reasoning
    edge = next step proposal
    leaf = answer

UCT 选 child: UCT(s,a) = Q(s,a) + c * sqrt(ln N(s) / N(s,a))
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class MCTSNode:
    state: str                                  # partial reasoning text
    parent: "MCTSNode | None" = None
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    is_terminal: bool = False
    score: float = 0.0                          # PRM step score

    @property
    def Q(self) -> float:
        return self.value_sum / max(self.visits, 1)

    def uct(self, c: float = 1.41) -> float:
        if self.visits == 0 or self.parent is None:
            return float("inf")
        return self.Q + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self, c: float = 1.41) -> "MCTSNode":
        return max(self.children, key=lambda n: n.uct(c))

    def add_child(self, state: str, prm_score: float = 0.5) -> "MCTSNode":
        node = MCTSNode(state=state, parent=self, score=prm_score)
        self.children.append(node)
        return node


def mcts_step(root: MCTSNode, expand_fn, simulate_fn, max_iter: int = 50):
    """单轮 MCTS."""
    for _ in range(max_iter):
        # 1. Selection
        node = root
        while node.children and not node.is_terminal:
            node = node.best_child()
        # 2. Expansion
        if not node.is_terminal:
            for new_state, prm_score in expand_fn(node.state):
                node.add_child(new_state, prm_score)
            if node.children:
                node = node.children[0]
        # 3. Simulation (rollout to terminal)
        reward = simulate_fn(node.state)
        # 4. Backprop
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value_sum += reward
            cur = cur.parent
    return root


def best_path(root: MCTSNode) -> list[str]:
    path = []
    node = root
    while node.children:
        node = max(node.children, key=lambda n: n.visits)
        path.append(node.state)
    return path


if __name__ == "__main__":
    print("MCTS for LLM (toy)\n" + "=" * 50)
    import random
    random.seed(7)

    def expand(state: str):
        """mock：每个 state 出 3 个候选 next step."""
        return [(state + f" step_{i}", random.uniform(0.2, 0.9)) for i in range(3)]

    def simulate(state: str) -> float:
        """mock：随机回报 + bias 偏向 step_2 (假装更好)."""
        if "step_2" in state:
            return random.uniform(0.6, 1.0)
        return random.uniform(0.1, 0.5)

    root = MCTSNode(state="Q: 1+2*3=?")
    final = mcts_step(root, expand, simulate, max_iter=30)
    path = best_path(final)
    print("最优路径:")
    for s in path:
        print(f"  → {s}")
    print(f"\n root visits = {final.visits}")
