"""HyDE — Hypothetical Document Embedding (Gao 2022)."""
from __future__ import annotations
from typing import Callable
from common import hash_embed, cosine, Doc, RetrievalResult
from naive_rag import NaiveRAG


def make_hypothesis_llm() -> Callable[[str], str]:
    """Mock LLM that elaborates a query into a passage-like hypothesis."""
    expansions = {
        "anthropic": "Anthropic is an AI safety company founded in 2021 by Dario and Daniela Amodei.",
        "graphrag": "GraphRAG is a Microsoft 2024 method using entity graphs and community summaries.",
        "react": "ReAct is an agent pattern with thought, action, observation cycles in 2022 by Yao.",
        "mcp": "MCP is Model Context Protocol introduced by Anthropic November 2024.",
        "bm25": "BM25 is a sparse retrieval scoring function from Robertson 1994 used widely.",
        "hybrid": "Hybrid retrieval combines BM25 and dense vectors using RRF fusion.",
        "hyde": "HyDE writes a hypothetical answer first then retrieves via its embedding.",
        "colbert": "ColBERT uses per-token embeddings with MaxSim late interaction Khattab 2020.",
        "ragas": "RAGAS measures faithfulness, answer relevancy, context precision and recall.",
        "hipporag": "HippoRAG uses personalized PageRank over entity graph for multi-hop QA.",
    }

    def llm(query: str) -> str:
        q_low = query.lower()
        for key, expansion in expansions.items():
            if key in q_low:
                return expansion
        return query + " — relevant 2024-2025 technique in retrieval-augmented generation."

    return llm


class HyDERAG:
    def __init__(self):
        self.base = NaiveRAG()
        self.llm = make_hypothesis_llm()

    def index(self, docs: list[Doc]) -> None:
        self.base.index(docs)

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        hypothesis = self.llm(query)
        return self.base.search(hypothesis, k=k)


def _self_test() -> None:
    from common import SAMPLE_DOCS

    rag = HyDERAG()
    rag.index(SAMPLE_DOCS)
    res = rag.search("Who founded Anthropic?", k=3)
    assert any("Anthropic" in r.chunk.text for r in res), [r.chunk.text for r in res]

    res = rag.search("What is HippoRAG?", k=3)
    assert any("HippoRAG" in r.chunk.text for r in res), [r.chunk.text for r in res]
    print("[OK] hyde_demo._self_test passed")


if __name__ == "__main__":
    _self_test()
