"""Multi-agent debate pattern."""
from __future__ import annotations
from typing import Callable
from dataclasses import dataclass


@dataclass
class DebateAgent:
    name: str
    answer_fn: Callable[[str], str]
    critique_fn: Callable[[str, list[dict]], str]


def debate(
    question: str,
    agents: list[DebateAgent],
    rounds: int = 3,
    judge: Callable[[str, list[dict]], dict] | None = None,
) -> dict:
    answers = [{"agent": a.name, "answer": a.answer_fn(question), "round": 1} for a in agents]
    history = list(answers)

    for r in range(2, rounds + 1):
        new_answers = []
        for a in agents:
            others = [ans for ans in answers if ans["agent"] != a.name]
            revised = a.critique_fn(question, others)
            new_answers.append({"agent": a.name, "answer": revised, "round": r})
        history.extend(new_answers)
        answers = new_answers

    if judge:
        verdict = judge(question, answers)
        return {"verdict": verdict, "history": history, "final_round": answers}
    return {"verdict": majority_vote(answers), "history": history, "final_round": answers}


def majority_vote(answers: list[dict]) -> dict:
    from collections import Counter
    c = Counter(ans["answer"] for ans in answers)
    best, count = c.most_common(1)[0]
    return {"best": best, "votes": count, "total": len(answers)}


def _self_test() -> None:
    a1 = DebateAgent(
        name="A",
        answer_fn=lambda q: "42",
        critique_fn=lambda q, others: "42",
    )
    a2 = DebateAgent(
        name="B",
        answer_fn=lambda q: "41",
        critique_fn=lambda q, others: "42" if any(o["answer"] == "42" for o in others) else "41",
    )
    a3 = DebateAgent(
        name="C",
        answer_fn=lambda q: "43",
        critique_fn=lambda q, others: "42" if sum(1 for o in others if o["answer"] == "42") >= 1 else "43",
    )

    res = debate("What is the answer?", [a1, a2, a3], rounds=3)
    assert res["verdict"]["best"] == "42"
    assert res["verdict"]["votes"] >= 2
    print("[OK] debate._self_test passed")


if __name__ == "__main__":
    _self_test()
