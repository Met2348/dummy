"""Toy Toolformer data filter.

This does not train a language model. It implements the paper's core
bookkeeping: linearize an API call, compare loss with and without the
tool result, keep calls whose result reduces future-token loss.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class APICandidate:
    position: int
    name: str
    argument: str
    result: str
    loss_without_call: float
    loss_call_only: float
    loss_with_result: float


def linearize_call(name: str, argument: str, result: str | None = None) -> str:
    if result is None:
        return f"<API> {name}({argument}) </API>"
    return f"<API> {name}({argument}) -> {result} </API>"


def toolformer_score(candidate: APICandidate) -> float:
    """Paper filter score: L_minus - L_plus."""
    l_minus = min(candidate.loss_without_call, candidate.loss_call_only)
    l_plus = candidate.loss_with_result
    return l_minus - l_plus


def keep_candidate(candidate: APICandidate, tau_f: float = 1.0) -> bool:
    return toolformer_score(candidate) >= tau_f


def filter_candidates(candidates: list[APICandidate], tau_f: float = 1.0) -> list[APICandidate]:
    return [c for c in candidates if keep_candidate(c, tau_f=tau_f)]


def interleave_call(tokens: list[str], candidate: APICandidate) -> list[str]:
    """Insert a kept API call before tokens[position]."""
    pos = max(0, min(candidate.position, len(tokens)))
    call = linearize_call(candidate.name, candidate.argument, candidate.result)
    return tokens[:pos] + [call] + tokens[pos:]


def _self_test() -> None:
    useful = APICandidate(
        position=4,
        name="QA",
        argument='"What other name is Pittsburgh known by?"',
        result="Steel City",
        loss_without_call=3.0,
        loss_call_only=2.8,
        loss_with_result=0.6,
    )
    bad = APICandidate(
        position=4,
        name="QA",
        argument='"Which country is Pittsburgh in?"',
        result="United States",
        loss_without_call=3.0,
        loss_call_only=2.9,
        loss_with_result=2.7,
    )

    assert round(toolformer_score(useful), 2) == 2.2
    assert keep_candidate(useful, tau_f=1.0)
    assert not keep_candidate(bad, tau_f=1.0)
    kept = filter_candidates([useful, bad], tau_f=1.0)
    assert kept == [useful]

    tokens = "Pittsburgh is also known as the Steel City .".split()
    augmented = interleave_call(tokens, useful)
    assert any("Steel City" in tok for tok in augmented)
    assert "<API>" in augmented[4]
    print("[OK] toolformer_toy._self_test passed")


if __name__ == "__main__":
    _self_test()
