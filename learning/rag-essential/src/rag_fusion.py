"""Multi-Query / RAG-Fusion — rewrite query N ways and RRF."""
from __future__ import annotations
from typing import Callable
from common import Doc, RetrievalResult
from hybrid import HybridRAG, rrf_fusion


def make_query_rewriter() -> Callable[[str], list[str]]:
    """Mock: expand query into 3 variants by simple keyword synonyms."""
    SYN = {
        "founded": ["created", "started", "established"],
        "made": ["built", "created", "developed"],
        "use": ["leverage", "employ"],
        "fast": ["quick", "rapid"],
    }

    def rewrite(query: str) -> list[str]:
        variants = [query]
        q_low = query.lower()
        for key, syns in SYN.items():
            if key in q_low:
                for s in syns:
                    variants.append(q_low.replace(key, s))
        if len(variants) == 1:
            words = query.split()
            if len(words) > 2:
                variants.append(" ".join(words[1:]))
                variants.append(" ".join(words[:-1]))
        return variants[:4]

    return rewrite


class RAGFusion:
    def __init__(self):
        self.base = HybridRAG()
        self.rewriter = make_query_rewriter()

    def index(self, docs: list[Doc]) -> None:
        self.base.index(docs)

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        variants = self.rewriter(query)
        per_query = [self.base.search(q, k=k * 2) for q in variants]
        if len(per_query) == 1:
            return per_query[0][:k]
        result = per_query[0]
        for nxt in per_query[1:]:
            result = rrf_fusion(result, nxt, k=k * 2)
        return result[:k]


def _self_test() -> None:
    from common import SAMPLE_DOCS

    r = RAGFusion()
    r.index(SAMPLE_DOCS)
    res = r.search("Who founded Anthropic?", k=3)
    assert len(res) > 0
    assert any("Anthropic" in x.chunk.text for x in res), [x.chunk.text for x in res]
    print("[OK] rag_fusion._self_test passed")


if __name__ == "__main__":
    _self_test()
