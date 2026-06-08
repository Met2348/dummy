"""Naive RAG: embed + cosine top-k."""
from __future__ import annotations
from common import Doc, hash_embed, cosine, RetrievalResult, Chunk
from chunker import sentence_chunk, to_chunk_objects


class NaiveRAG:
    def __init__(self, embed_fn=hash_embed):
        self.embed_fn = embed_fn
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []

    def index(self, docs: list[Doc]) -> None:
        for d in docs:
            for ch in to_chunk_objects(d.id, sentence_chunk(d.text)):
                self.chunks.append(ch)
                self.vectors.append(self.embed_fn(ch.text))

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        q_vec = self.embed_fn(query)
        scored = [
            RetrievalResult(chunk=ch, score=cosine(q_vec, v))
            for ch, v in zip(self.chunks, self.vectors)
        ]
        return sorted(scored, key=lambda r: r.score, reverse=True)[:k]


def _self_test() -> None:
    from common import SAMPLE_DOCS, SAMPLE_QUERIES

    rag = NaiveRAG()
    rag.index(SAMPLE_DOCS)
    assert len(rag.chunks) >= len(SAMPLE_DOCS)

    q, expected = SAMPLE_QUERIES[0]  # "Who founded Anthropic?"
    results = rag.search(q, k=5)
    assert len(results) == 5
    # At least one result should mention Anthropic.
    assert any("Anthropic" in r.chunk.text for r in results), \
        [r.chunk.text for r in results]
    print("[OK] naive_rag._self_test passed")


if __name__ == "__main__":
    _self_test()
