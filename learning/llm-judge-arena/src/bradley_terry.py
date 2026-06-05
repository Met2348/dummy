"""Bradley-Terry model + Elo aggregation.

BT model: P(A beats B) = exp(r_A) / (exp(r_A) + exp(r_B))
    log P(win) = r_A - logsumexp(r_A, r_B)

Maximum-likelihood fit via iterative method (Hunter 2004 MM algorithm).
Equivalent to Chatbot Arena's Elo rating with logistic loss.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from common import PairBattle


def fit_bt(battles: List[PairBattle], n_iter: int = 100) -> Dict[str, float]:
    """Fit Bradley-Terry strengths via MM algorithm.

    Returns dict {model_name: log_strength} centered around 0.
    """
    models = sorted({b.model_a for b in battles} | {b.model_b for b in battles})
    if not models:
        return {}
    # Initialize at log-strength 0
    log_s = {m: 0.0 for m in models}
    # Build win counts (tie = 0.5 win each)
    W: Dict[Tuple[str, str], float] = {}
    for b in battles:
        if b.winner == "A":
            W[(b.model_a, b.model_b)] = W.get((b.model_a, b.model_b), 0) + 1
        elif b.winner == "B":
            W[(b.model_b, b.model_a)] = W.get((b.model_b, b.model_a), 0) + 1
        else:
            W[(b.model_a, b.model_b)] = W.get((b.model_a, b.model_b), 0) + 0.5
            W[(b.model_b, b.model_a)] = W.get((b.model_b, b.model_a), 0) + 0.5

    # MM updates: s_i ← W_i / Σ_j (N_{ij} / (s_i + s_j))
    for _ in range(n_iter):
        s = {m: math.exp(log_s[m]) for m in models}
        new_s = {}
        for i in models:
            W_i = sum(W.get((i, j), 0) for j in models if j != i)
            denom = 0.0
            for j in models:
                if i == j:
                    continue
                n_ij = W.get((i, j), 0) + W.get((j, i), 0)
                if n_ij > 0:
                    denom += n_ij / (s[i] + s[j])
            new_s[i] = (W_i / denom) if denom > 0 else s[i]
        log_s = {m: math.log(max(1e-9, new_s[m])) for m in models}
        # Center to mean 0
        mean = sum(log_s.values()) / len(log_s)
        log_s = {m: v - mean for m, v in log_s.items()}
    return log_s


def to_elo(log_strengths: Dict[str, float], base: int = 1500, scale: float = 400/math.log(10)) -> Dict[str, int]:
    """Convert BT log-strength to Elo (1500 base, 400 per 10x odds)."""
    return {m: int(base + scale * v) for m, v in log_strengths.items()}


def _self_test() -> int:
    # 3 models: A always beats B, B always beats C
    battles = []
    for _ in range(20):
        battles.append(PairBattle("q", "A", "B", "A"))
        battles.append(PairBattle("q", "B", "C", "A"))
    bt = fit_bt(battles, n_iter=200)
    # A should have highest strength, C lowest
    assert bt["A"] > bt["B"] > bt["C"], bt
    elo = to_elo(bt)
    assert elo["A"] > elo["B"] > elo["C"], elo
    # Range plausible
    assert 100 < elo["A"] - elo["C"]
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"bradley_terry.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
