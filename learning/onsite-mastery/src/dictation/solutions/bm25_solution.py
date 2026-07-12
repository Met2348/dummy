"""bm25_spec 的参考实现。纯 stdlib，经典 Robertson-Sparck-Jones IDF 形式（不做 +1 平滑）。"""
from __future__ import annotations

import math
from collections import Counter


def bm25_score(
    query_terms: list[str],
    doc_terms: list[str],
    corpus_doc_freqs: dict[str, int],
    corpus_size: int,
    avg_doc_len: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    doc_len = len(doc_terms)
    term_counts = Counter(doc_terms)
    length_norm = 1.0 - b + b * (doc_len / avg_doc_len)

    score = 0.0
    for t in query_terms:
        n_t = corpus_doc_freqs.get(t, 0)
        idf = math.log((corpus_size - n_t + 0.5) / (n_t + 0.5))
        f_td = term_counts.get(t, 0)
        tf_component = f_td * (k1 + 1) / (f_td + k1 * length_norm)
        score += idf * tf_component
    return score
