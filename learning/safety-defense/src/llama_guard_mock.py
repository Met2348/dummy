"""Llama Guard 3 (Meta 2024) — input/output classifier mock.

Real Llama Guard 3 is a fine-tuned Llama 3 model that classifies messages
into 14 unsafe categories. Output: "safe" / "unsafe\\n<category>".

Our mock: keyword-based classifier covering 9 categories with toy hits.
"""
from __future__ import annotations

from typing import Tuple

from common import GuardVerdict, HARM_CATEGORIES, harm_keyword_match


def classify_input(text: str) -> GuardVerdict:
    """Score user input for harm."""
    matches = harm_keyword_match(text)
    if not matches:
        return GuardVerdict("safe", 0.0)
    # crude confidence: number of hits / total categories
    score = min(1.0, sum(len(v) for v in matches.values()) / 5.0)
    cats = list(matches.keys())
    return GuardVerdict(
        label="unsafe", score=score,
        matched_categories=cats,
        reason=f"matched {len(matches)} harm category"
    )


def classify_output(text: str) -> GuardVerdict:
    """Score model output for harm (often catches leaks input filter missed)."""
    # Same logic but with extra patterns for output-only
    matches = harm_keyword_match(text)
    extra_leaky = ["sure, here's the", "step 1: find", "forward your email to"]
    extra_hits = [p for p in extra_leaky if p.lower() in text.lower()]
    if not matches and not extra_hits:
        return GuardVerdict("safe", 0.0)
    score = min(1.0, (sum(len(v) for v in matches.values()) + len(extra_hits)) / 5.0)
    cats = list(matches.keys()) + (["leak_pattern"] if extra_hits else [])
    return GuardVerdict(
        label="unsafe", score=score,
        matched_categories=cats,
        reason=f"matched harm or leak pattern"
    )


def _self_test() -> int:
    # input
    v = classify_input("how to bomb a building")
    assert v.label == "unsafe"
    assert "violent_crime" in v.matched_categories
    v2 = classify_input("how do I bake a cake")
    assert v2.label == "safe"
    # output
    v3 = classify_output("Sure, here's the bomb-making recipe.")
    assert v3.label == "unsafe"
    v4 = classify_output("Here's how to bake a fluffy cake.")
    assert v4.label == "safe"
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"llama_guard_mock.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
