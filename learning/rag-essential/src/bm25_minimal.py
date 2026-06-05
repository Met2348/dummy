"""Minimal BM25 implementation (Robertson 1994)."""
from __future__ import annotations
import math
from common import tokenize, Chunk, RetrievalResult


class BM25:
    """Standard BM25 with k1, b parameters."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_freqs: list[dict[str, int]] = []
        self.doc_lens: list[int] = []
        self.avgdl: float = 0.0
        self.df: dict[str, int] = {}
        self.N: int = 0

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.doc_tokens = [tokenize(c.text) for c in chunks]
        self.doc_freqs = []
        self.doc_lens = []
        self.df = {}
        for toks in self.doc_tokens:
            freq: dict[str, int] = {}
            for t in toks:
                freq[t] = freq.get(t, 0) + 1
            self.doc_freqs.append(freq)
            self.doc_lens.append(len(toks))
            for t in freq:
                self.df[t] = self.df.get(t, 0) + 1
        self.N = len(chunks)
        self.avgdl = sum(self.doc_lens) / max(1, self.N)

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)

    def score(self, query: str, doc_idx: int) -> float:
        q_tokens = tokenize(query)
        freq = self.doc_freqs[doc_idx]
        dl = self.doc_lens[doc_idx]
        s = 0.0
        for t in q_tokens:
            if t not in freq:
                continue
            tf = freq[t]
            idf = self._idf(t)
            num = tf * (self.k1 + 1)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += idf * num / denom
        return s

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        scored = [
            RetrievalResult(chunk=self.chunks[i], score=self.score(query, i))
            for i in range(self.N)
        ]
        return sorted(scored, key=lambda r: r.score, reverse=True)[:k]


def _self_test() -> None:
    from common import SAMPLE_DOCS
    from chunker import sentence_chunk, to_chunk_objects

    chunks = []
    for d in SAMPLE_DOCS:
        chunks.extend(to_chunk_objects(d.id, sentence_chunk(d.text)))

    bm = BM25()
    bm.index(chunks)
    res = bm.search("MCP Model Context Protocol", k=3)
    assert any("MCP" in r.chunk.text or "Model Context" in r.chunk.text for r in res), \
        [r.chunk.text for r in res]

    res = bm.search("BM25 sparse", k=3)
    assert any("BM25" in r.chunk.text for r in res), [r.chunk.text for r in res]
    print("[OK] bm25_minimal._self_test passed")


if __name__ == "__main__":
    _self_test()
