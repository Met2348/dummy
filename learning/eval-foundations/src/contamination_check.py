"""Benchmark contamination detection (n-gram + canary).

Two classic methods:
1. N-gram overlap (Carlini 2021, Brown 2020): check if bench substrings appear in training corpus
2. Canary string (Sainz 2023): embed unique IDs in test set, search corpus

Plus a Min-K%++ (Shi 2024) sketch — uses per-token log-prob distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class ContaminationReport:
    method: str
    benchmark: str
    overlap_score: float
    flagged: bool
    detail: str


def _ngrams(text: str, n: int) -> Set[str]:
    """Word-level n-grams."""
    words = text.lower().split()
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


def ngram_overlap(bench_texts: List[str], corpus_texts: List[str],
                   n: int = 13, threshold: float = 0.5) -> ContaminationReport:
    """13-gram overlap (Brown 2020 GPT-3 paper standard)."""
    bench_grams = set()
    for t in bench_texts:
        bench_grams |= _ngrams(t, n)
    corpus_grams = set()
    for t in corpus_texts:
        corpus_grams |= _ngrams(t, n)
    if not bench_grams:
        return ContaminationReport("ngram", "?", 0.0, False, "empty bench")
    overlap = len(bench_grams & corpus_grams) / len(bench_grams)
    return ContaminationReport(
        method=f"{n}-gram",
        benchmark="?",
        overlap_score=overlap,
        flagged=(overlap >= threshold),
        detail=f"{len(bench_grams & corpus_grams)}/{len(bench_grams)} grams in corpus",
    )


def canary_search(bench_texts: List[str], corpus_texts: List[str],
                   canary: str) -> ContaminationReport:
    """Look for unique canary string in corpus.

    Real implementation: bench creators embed a unique ID like 'BIG-bench
    canary GUID 12345' that should never appear naturally.
    """
    n_hits = sum(1 for t in corpus_texts if canary in t)
    return ContaminationReport(
        method="canary",
        benchmark="?",
        overlap_score=float(n_hits),
        flagged=(n_hits > 0),
        detail=f"canary '{canary}' found in {n_hits} corpus docs",
    )


def min_k_pct_pp_sketch(per_token_logp: List[float], k_pct: float = 20.0) -> float:
    """Min-K%++ proxy: avg log-prob of the lowest k% tokens.

    Real method also normalizes by token-level mean & std. Here we sketch
    the core intuition: contaminated samples have unusually uniform
    high log-probs (no surprises).
    """
    if not per_token_logp:
        return 0.0
    sorted_lp = sorted(per_token_logp)
    n_lowest = max(1, int(len(sorted_lp) * k_pct / 100))
    return sum(sorted_lp[:n_lowest]) / n_lowest


def _self_test() -> int:
    # ngram overlap
    bench = ["the cat sat on the mat in the bright morning",
             "what is the capital of France"]
    clean_corpus = ["completely different text here for testing"]
    contam_corpus = ["the cat sat on the mat in the bright morning"]
    r_clean = ngram_overlap(bench, clean_corpus, n=5, threshold=0.5)
    assert not r_clean.flagged
    r_dirty = ngram_overlap(bench, contam_corpus, n=5, threshold=0.5)
    assert r_dirty.overlap_score > 0.0
    # canary
    r_c = canary_search(bench, ["GUID-CANARY-42 was found", "other"], "GUID-CANARY-42")
    assert r_c.flagged
    # min-k
    lp = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0]
    score = min_k_pct_pp_sketch(lp, k_pct=20.0)
    assert score == -9.5  # avg of -9, -10
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"contamination_check.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
