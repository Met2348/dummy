"""Original RAG probability toy: latent-document marginalization."""
from __future__ import annotations

import math


def normalize(scores: dict[str, float]) -> dict[str, float]:
    total = sum(scores.values())
    if total <= 0:
        raise ValueError("scores must sum to a positive value")
    return {k: v / total for k, v in scores.items()}


def rag_sequence_prob(
    doc_probs: dict[str, float],
    token_probs_by_doc: dict[str, list[float]],
) -> float:
    """RAG-Sequence: one latent document explains the whole output."""
    pz = normalize(doc_probs)
    total = 0.0
    for doc_id, doc_prob in pz.items():
        token_probs = token_probs_by_doc[doc_id]
        seq_prob = math.prod(token_probs)
        total += doc_prob * seq_prob
    return total


def rag_token_prob(
    doc_probs: dict[str, float],
    token_probs_by_doc: dict[str, list[float]],
) -> float:
    """RAG-Token: marginalize documents separately for each output token."""
    pz = normalize(doc_probs)
    n_tokens = len(next(iter(token_probs_by_doc.values())))
    total = 1.0
    for token_idx in range(n_tokens):
        token_total = 0.0
        for doc_id, doc_prob in pz.items():
            token_total += doc_prob * token_probs_by_doc[doc_id][token_idx]
        total *= token_total
    return total


def dpr_inner_product(query_vec: list[float], doc_vec: list[float]) -> float:
    if len(query_vec) != len(doc_vec):
        raise ValueError("dimension mismatch")
    return sum(q * d for q, d in zip(query_vec, doc_vec))


def top_k_docs(
    query_vec: list[float],
    doc_vectors: dict[str, list[float]],
    k: int,
) -> list[tuple[str, float]]:
    scored = [
        (doc_id, dpr_inner_product(query_vec, doc_vec))
        for doc_id, doc_vec in doc_vectors.items()
    ]
    return sorted(scored, key=lambda item: item[1], reverse=True)[:k]


def hot_swap_answer(query: str, index: dict[str, str]) -> str:
    """Tiny stand-in for replacing RAG's non-parametric memory index."""
    q = query.lower()
    for key, answer in index.items():
        if key.lower() in q:
            return answer
    return "unknown"


def _self_test() -> None:
    doc_probs = {"d1": 0.7, "d2": 0.3}
    token_probs_by_doc = {
        "d1": [0.9, 0.8],
        "d2": [0.1, 0.95],
    }
    seq = rag_sequence_prob(doc_probs, token_probs_by_doc)
    tok = rag_token_prob(doc_probs, token_probs_by_doc)
    assert round(seq, 4) == 0.5325, seq
    assert round(tok, 4) == 0.5577, tok
    assert tok > seq

    docs = {
        "a": [1.0, 0.0, 0.0],
        "b": [0.1, 0.9, 0.0],
        "c": [0.0, 0.0, 1.0],
    }
    assert top_k_docs([1.0, 0.0, 0.0], docs, k=2)[0][0] == "a"

    old_index = {"capital of exampleland": "Oldtown"}
    new_index = {"capital of exampleland": "Newcity"}
    assert hot_swap_answer("What is the capital of Exampleland?", old_index) == "Oldtown"
    assert hot_swap_answer("What is the capital of Exampleland?", new_index) == "Newcity"

    print("[OK] rag_original_minimal._self_test passed")


if __name__ == "__main__":
    _self_test()
