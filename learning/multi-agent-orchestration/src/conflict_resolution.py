"""Conflict resolution: vote / weighted / judge / borda."""
from __future__ import annotations
from collections import Counter
from typing import Callable


def majority_vote(answers: list[str]) -> dict:
    if not answers:
        return {"best": None, "votes": 0, "total": 0}
    c = Counter(answers)
    best, count = c.most_common(1)[0]
    return {"best": best, "votes": count, "total": len(answers), "counts": dict(c)}


def weighted_vote(answers: list[str], weights: list[float]) -> dict:
    if len(answers) != len(weights):
        raise ValueError("answers/weights length mismatch")
    scores: dict[str, float] = {}
    for a, w in zip(answers, weights):
        scores[a] = scores.get(a, 0.0) + w
    best = max(scores, key=scores.get)
    return {"best": best, "score": scores[best], "scores": scores}


def vote_with_tie_break(answers: list[str], fallback: str = "ABSTAIN") -> dict:
    if not answers:
        return {"best": fallback, "tie": False}
    c = Counter(answers)
    top_two = c.most_common(2)
    if len(top_two) >= 2 and top_two[0][1] == top_two[1][1]:
        return {"best": fallback, "tie": True, "candidates": [t[0] for t in top_two]}
    return {"best": top_two[0][0], "tie": False, "counts": dict(c)}


def borda_count(rankings: list[list[str]]) -> dict:
    """Each ranking is a list of candidates best-first. Returns winner."""
    scores: dict[str, int] = {}
    for ranking in rankings:
        n = len(ranking)
        for i, cand in enumerate(ranking):
            scores[cand] = scores.get(cand, 0) + (n - i)
    winner = max(scores, key=scores.get)
    return {"best": winner, "scores": scores}


def judge_pick(
    question: str,
    answers: list[str],
    judge_fn: Callable[[str, list[str]], int],
) -> dict:
    idx = judge_fn(question, answers)
    return {"best": answers[idx], "judge_idx": idx}


def _self_test() -> None:
    r = majority_vote(["A", "A", "B", "A", "C"])
    assert r["best"] == "A" and r["votes"] == 3

    r = weighted_vote(["A", "B", "A"], [0.3, 0.9, 0.3])
    assert r["best"] == "B", r

    r = vote_with_tie_break(["A", "B", "A", "B"])
    assert r["tie"] and r["best"] == "ABSTAIN"

    r = vote_with_tie_break(["A", "A", "B"])
    assert r["best"] == "A" and not r["tie"]

    r = borda_count([["A", "B", "C"], ["B", "A", "C"], ["A", "C", "B"]])
    assert r["best"] == "A", r

    r = judge_pick("Q", ["wrong", "right"], lambda q, ans: 1)
    assert r["best"] == "right"
    print("[OK] conflict_resolution._self_test passed")


if __name__ == "__main__":
    _self_test()
