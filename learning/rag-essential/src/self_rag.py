"""CRAG-style RAG with confidence-branching."""
from __future__ import annotations
from typing import Callable
from common import Doc, RetrievalResult, tokenize, Chunk
from hybrid import HybridRAG
from reranker_mock import keyword_overlap_score


def confidence_score(query: str, results: list[RetrievalResult]) -> float:
    if not results:
        return 0.0
    top_score = max(keyword_overlap_score(query, r.chunk.text) for r in results[:3])
    return top_score


def mock_web_search(query: str) -> list[RetrievalResult]:
    """Mock web fallback — returns a single canned 'web' result."""
    snippet = f"[web] Relevant 2025 result for: {query}"
    ch = Chunk(doc_id="web", chunk_id="web1", text=snippet)
    return [RetrievalResult(chunk=ch, score=0.5)]


class CRAG:
    def __init__(self):
        self.base = HybridRAG()
        self.web = mock_web_search

    def index(self, docs: list[Doc]) -> None:
        self.base.index(docs)

    def search(
        self,
        query: str,
        k: int = 5,
        high_thresh: float = 0.5,
        low_thresh: float = 0.15,
    ) -> tuple[str, list[RetrievalResult]]:
        results = self.base.search(query, k=k)
        conf = confidence_score(query, results)
        if conf >= high_thresh:
            return ("high", results)
        elif conf >= low_thresh:
            return ("mid", results + self.web(query))
        else:
            return ("low", self.web(query))


def _self_test() -> None:
    from common import SAMPLE_DOCS

    crag = CRAG()
    crag.index(SAMPLE_DOCS)

    branch, res = crag.search("Anthropic Claude founded", k=3)
    assert branch in ("high", "mid", "low"), branch
    assert len(res) > 0

    branch_low, res_low = crag.search("completely unrelated extraterrestrial xyzxyz", k=3)
    assert branch_low == "low", (branch_low, [r.chunk.text for r in res_low])
    assert any("[web]" in r.chunk.text for r in res_low)
    print("[OK] self_rag._self_test passed")


if __name__ == "__main__":
    _self_test()
