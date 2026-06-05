"""Judge bias reproductions — position / verbosity / self-pref / style.

We construct synthetic battles where the "true" winner is known, then
swap orderings / lengths to expose bias.
"""
from __future__ import annotations

from typing import Dict, List

from common import PairBattle, PairJudgeFn, make_length_judge


def position_bias_score(judge: PairJudgeFn, n_pairs: int = 20) -> Dict:
    """How often does the judge prefer position A regardless of content?

    True answer should be 50/50 (random tie).
    """
    same_text = "this is the same answer in both positions"
    a_wins = 0; b_wins = 0; ties = 0
    for i in range(n_pairs):
        q = f"question {i}"
        v = judge(q, same_text, same_text)
        if v == "A":
            a_wins += 1
        elif v == "B":
            b_wins += 1
        else:
            ties += 1
    return {"a_wins": a_wins, "b_wins": b_wins, "ties": ties,
            "position_skew": (a_wins - b_wins) / n_pairs}


def verbosity_bias_score(judge: PairJudgeFn, n_pairs: int = 20) -> Dict:
    """Does the judge favor the verbose response (longer = better)?

    True: length should not matter. We use a judge that knows it.
    """
    a_wins = 0
    for i in range(n_pairs):
        q = f"question {i}"
        short = "answer."
        long = "this is a much longer answer with more words but no extra info."
        v = judge(q, short, long)
        if v == "B":
            a_wins += 1  # 'B' = long here
    return {"long_pick_rate": a_wins / n_pairs}


def swap_consistency(judge: PairJudgeFn, n_pairs: int = 20) -> float:
    """If we swap A and B, the winner should also swap (consistency).

    Real judges score ~60-80% consistency.
    """
    consistent = 0
    for i in range(n_pairs):
        q = f"q {i}"
        a, b = f"answer A {i}", f"answer B {i} (long)"
        v1 = judge(q, a, b)
        v2 = judge(q, b, a)
        # If v1='A' then v2 should be 'B', and vice versa
        if (v1, v2) in (("A", "B"), ("B", "A"), ("tie", "tie")):
            consistent += 1
    return consistent / n_pairs


def _self_test() -> int:
    # Length judge: prefers longer → 100% verbosity bias
    j_long = make_length_judge(prefer_longer=True)
    vb = verbosity_bias_score(j_long)
    assert vb["long_pick_rate"] == 1.0
    # Position bias score: same text → length tie, so 100% tie
    pb = position_bias_score(j_long)
    assert pb["ties"] == 20
    # Swap consistency: length judge is consistent
    sc = swap_consistency(j_long, n_pairs=20)
    assert sc == 1.0
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"judge_bias_demo.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
