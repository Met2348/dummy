"""Llama Guard input/output classifier mock.

Real Llama Guard is a fine-tuned Llama model that classifies messages
with a policy taxonomy. Output: "safe" or "unsafe\\n<category>".

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
    extra_leaky = ["unsafe-output-pattern", "forward your email to"]
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
    v = classify_input("unsafe:violence_hate evaluation item")
    assert v.label == "unsafe"
    assert "violence_hate" in v.matched_categories
    v2 = classify_input("how do I bake a cake")
    assert v2.label == "safe"
    # output
    v3 = classify_output("unsafe-output-pattern: toy unsafe response")
    assert v3.label == "unsafe"
    v4 = classify_output("Here's how to bake a fluffy cake.")
    assert v4.label == "safe"
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"llama_guard_mock.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
