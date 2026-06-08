"""Minimal ColBERT - per-token embeddings plus MaxSim aggregation."""
from __future__ import annotations
from common import hash_embed, cosine, tokenize, Chunk, RetrievalResult, Doc
from chunker import sentence_chunk, to_chunk_objects


def per_token_embed(text: str, dim: int = 32) -> list[list[float]]:
    return [hash_embed(t, dim=dim) for t in tokenize(text)]


def maxsim_score(query_vecs: list[list[float]], doc_vecs: list[list[float]]) -> float:
    if not query_vecs or not doc_vecs:
        return 0.0
    s = 0.0
    for qv in query_vecs:
        s += max(cosine(qv, dv) for dv in doc_vecs)
    return s / len(query_vecs)


class ColBERTLite:
    def __init__(self, dim: int = 32):
        self.dim = dim
        self.chunks: list[Chunk] = []
        self.token_vecs: list[list[list[float]]] = []

    def index(self, docs: list[Doc]) -> None:
        for d in docs:
            for ch in to_chunk_objects(d.id, sentence_chunk(d.text)):
                self.chunks.append(ch)
                self.token_vecs.append(per_token_embed(ch.text, dim=self.dim))

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        q_vecs = per_token_embed(query, dim=self.dim)
        scored = [
            RetrievalResult(chunk=self.chunks[i], score=maxsim_score(q_vecs, self.token_vecs[i]))
            for i in range(len(self.chunks))
        ]
        return sorted(scored, key=lambda r: r.score, reverse=True)[:k]


def _self_test() -> None:
    from common import SAMPLE_DOCS

    cb = ColBERTLite()
    cb.index(SAMPLE_DOCS)
    res = cb.search("ColBERT MaxSim per-token", k=3)
    assert any("ColBERT" in r.chunk.text or "MaxSim" in r.chunk.text for r in res), \
        [r.chunk.text for r in res]
    print("[OK] colbert_minimal._self_test passed")


if __name__ == "__main__":
    _self_test()
