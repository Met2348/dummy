"""Shared utilities for llm-judge-arena Topic 4."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# Generator: prompt -> response text
GenFn = Callable[[str], str]
# Judge: (question, response_a, response_b) -> 'A' | 'B' | 'tie'
PairJudgeFn = Callable[[str, str, str], str]
# Pointwise judge: (question, response) -> int score (e.g. 1-5)
PointJudgeFn = Callable[[str, str], int]


@dataclass
class PairBattle:
    qid: str
    model_a: str
    model_b: str
    winner: str   # 'A' / 'B' / 'tie'


@dataclass
class Sample:
    qid: str
    prompt: str
    category: Optional[str] = None
    meta: Dict = field(default_factory=dict)


# === Bench data ===

MT_BENCH_QS: List[Sample] = [
    Sample("mt_1", "Explain quantum entanglement to a 12-year-old.", "knowledge"),
    Sample("mt_2", "Write a haiku about autumn.", "writing"),
    Sample("mt_3", "Solve 2x + 5 = 17 step by step.", "reasoning"),
    Sample("mt_4", "Translate 'Hello, world!' into French and German.", "linguistics"),
    Sample("mt_5", "Suggest 3 ways to reduce screen time.", "advice"),
    Sample("mt_6", "Continue the conversation: User says 'I'm sad'.", "roleplay"),
    Sample("mt_7", "Write a Python function to reverse a list.", "coding"),
    Sample("mt_8", "What are the pros/cons of remote work?", "open-ended"),
]


# === Generator helpers ===

def make_fixed_gen(answers: Dict[str, str], default: str = "(no response)") -> GenFn:
    """Generator that returns canned answers per qid."""
    def _fn(prompt: str) -> str:
        for k, v in answers.items():
            if f"[qid={k}]" in prompt:
                return v
        return default
    return _fn


# === Judge helpers ===

def make_length_judge(prefer_longer: bool = True) -> PairJudgeFn:
    """Toy 'judge' that just picks the longer (or shorter) response.

    Demonstrates verbosity bias (real judges have this issue).
    """
    def _fn(q: str, a: str, b: str) -> str:
        la, lb = len(a), len(b)
        if la == lb:
            return "tie"
        if prefer_longer:
            return "A" if la > lb else "B"
        return "A" if la < lb else "B"
    return _fn


def make_keyword_judge(keywords: List[str]) -> PairJudgeFn:
    """Judge that counts keyword matches."""
    def _fn(q: str, a: str, b: str) -> str:
        ca = sum(1 for k in keywords if k.lower() in a.lower())
        cb = sum(1 for k in keywords if k.lower() in b.lower())
        if ca > cb:
            return "A"
        if cb > ca:
            return "B"
        return "tie"
    return _fn


def make_position_biased_judge(base: PairJudgeFn, bias: float = 0.0) -> PairJudgeFn:
    """Wrap a judge with positional bias (flip 'B' to 'A' with prob `bias`).

    Demonstrates that judges often prefer position A.
    """
    def _fn(q: str, a: str, b: str) -> str:
        v = base(q, a, b)
        # deterministic 'bias' by hashing prompt
        import hashlib
        h = int(hashlib.md5(q.encode()).hexdigest(), 16) / (16**32)
        if v == "B" and h < bias:
            return "A"
        return v
    return _fn


def _self_test() -> int:
    # length judge
    j = make_length_judge(prefer_longer=True)
    assert j("q", "a", "longer answer") == "B"
    assert j("q", "long answer", "x") == "A"
    assert j("q", "abc", "xyz") == "tie"
    # keyword judge
    j2 = make_keyword_judge(["good", "great"])
    assert j2("q", "good", "bad") == "A"
    assert j2("q", "bad", "great") == "B"
    # gen helper
    g = make_fixed_gen({"q1": "alpha", "q2": "beta"})
    assert g("[qid=q1] ...") == "alpha"
    assert g("[qid=q3] ...") == "(no response)"
    # position bias
    j3 = make_position_biased_judge(j, bias=0.5)
    # outcome depends on hash, just test callable
    assert j3("q", "x", "longer") in ("A", "B", "tie")
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"common.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
