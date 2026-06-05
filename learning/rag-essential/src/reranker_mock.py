"""Mock reranker — token-overlap weighted cross-encoder simulation."""
from __future__ import annotations
from common import RetrievalResult, tokenize


def keyword_overlap_score(query: str, doc: str) -> float:
    q_tokens = set(tokenize(query))
    d_tokens = set(tokenize(doc))
    if not q_tokens:
        return 0.0
    overlap = len(q_tokens & d_tokens)
    return overlap / len(q_tokens)


def rerank(
    query: str,
    candidates: list[RetrievalResult],
    k: int = 5,
) -> list[RetrievalResult]:
    rescored = [
        RetrievalResult(chunk=r.chunk, score=keyword_overlap_score(query, r.chunk.text))
        for r in candidates
    ]
    return sorted(rescored, key=lambda r: r.score, reverse=True)[:k]


def _self_test() -> None:
    from common import SAMPLE_DOCS
    from hybrid import HybridRAG

    h = HybridRAG()
    h.index(SAMPLE_DOCS)
    candidates = h.search("HippoRAG personalized PageRank", k=10)
    ranked = rerank("HippoRAG personalized PageRank", candidates, k=3)
    assert len(ranked) == 3
    assert "HippoRAG" in ranked[0].chunk.text or "PageRank" in ranked[0].chunk.text, ranked[0].chunk.text
    print("[OK] reranker_mock._self_test passed")


if __name__ == "__main__":
    _self_test()
