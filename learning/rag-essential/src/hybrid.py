"""Hybrid retrieval: BM25 + dense + RRF fusion."""
from __future__ import annotations
from common import Doc, Chunk, RetrievalResult
from chunker import sentence_chunk, to_chunk_objects
from naive_rag import NaiveRAG
from bm25_minimal import BM25


class HybridRAG:
    def __init__(self):
        self.dense = NaiveRAG()
        self.bm25 = BM25()
        self.chunks: list[Chunk] = []

    def index(self, docs: list[Doc]) -> None:
        self.dense.index(docs)
        self.chunks = []
        for d in docs:
            self.chunks.extend(to_chunk_objects(d.id, sentence_chunk(d.text)))
        self.bm25.index(self.chunks)

    def search(self, query: str, k: int = 5, rrf_k: int = 60) -> list[RetrievalResult]:
        dense_results = self.dense.search(query, k=k * 4)
        bm25_results = self.bm25.search(query, k=k * 4)
        return rrf_fusion(dense_results, bm25_results, k=k, rrf_k=rrf_k)


def rrf_fusion(
    list_a: list[RetrievalResult],
    list_b: list[RetrievalResult],
    k: int,
    rrf_k: int = 60,
) -> list[RetrievalResult]:
    """Reciprocal Rank Fusion. Returns top-k."""
    scores: dict[str, float] = {}
    by_id: dict[str, RetrievalResult] = {}

    for rank, r in enumerate(list_a, start=1):
        cid = r.chunk.chunk_id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        by_id[cid] = r

    for rank, r in enumerate(list_b, start=1):
        cid = r.chunk.chunk_id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        if cid not in by_id:
            by_id[cid] = r

    top_ids = sorted(scores, key=scores.get, reverse=True)[:k]
    return [RetrievalResult(chunk=by_id[cid].chunk, score=scores[cid]) for cid in top_ids]


def _self_test() -> None:
    from common import SAMPLE_DOCS

    h = HybridRAG()
    h.index(SAMPLE_DOCS)

    res = h.search("Who founded Anthropic", k=5)
    assert len(res) == 5
    assert any("Anthropic" in r.chunk.text for r in res), [r.chunk.text for r in res]

    res = h.search("ColBERT MaxSim", k=3)
    assert any("ColBERT" in r.chunk.text for r in res), [r.chunk.text for r in res]
    print("[OK] hybrid._self_test passed")


if __name__ == "__main__":
    _self_test()
