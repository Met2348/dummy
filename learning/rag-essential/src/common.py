"""Common dataclasses & helpers for rag-essential."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import math
import re


@dataclass
class Doc:
    id: str
    text: str
    meta: dict = field(default_factory=dict)


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


def hash_embed(text: str, dim: int = 64) -> list[float]:
    """Deterministic bag-of-words hash embedding."""
    vec = [0.0] * dim
    for tok in re.findall(r"\w+", text.lower()):
        vec[hash(tok) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("dim mismatch")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


SAMPLE_DOCS: list[Doc] = [
    Doc("d01", "Claude is an AI assistant made by Anthropic, founded in 2021."),
    Doc("d02", "ReAct is an agent pattern: thought, action, observation loop."),
    Doc("d03", "GraphRAG by Microsoft 2024 uses entity graph and community summary."),
    Doc("d04", "HippoRAG uses personalized PageRank over entity graph for multi-hop QA."),
    Doc("d05", "MCP, the Model Context Protocol, was introduced by Anthropic in November 2024."),
    Doc("d06", "BM25 is a classic sparse retrieval scoring function from Robertson 1994."),
    Doc("d07", "Hybrid retrieval combines BM25 and dense vectors via RRF fusion."),
    Doc("d08", "Reranker is a cross-encoder that scores (query, doc) pairs."),
    Doc("d09", "ColBERT by Khattab 2020 uses late interaction MaxSim of per-token embeddings."),
    Doc("d10", "HyDE asks LLM to write a hypothetical answer first, then retrieves by that embedding."),
    Doc("d11", "RAGAS evaluates RAG using faithfulness, answer relevancy, context precision and recall."),
    Doc("d12", "Self-RAG by Asai 2023 uses special tokens to self-evaluate retrieval need and quality."),
    Doc("d13", "CRAG corrects RAG with three confidence branches and optional web search fallback."),
    Doc("d14", "Voyage-3 by Voyage AI is the recommended embedding model by Anthropic in 2024."),
    Doc("d15", "Cohere rerank-3 is a state-of-the-art commercial reranker model."),
    Doc("d16", "BGE-large-en-v1.5 by BAAI is the most popular open-source embedding model."),
    Doc("d17", "Matryoshka embeddings can be truncated while keeping retrieval quality."),
    Doc("d18", "Louvain algorithm finds communities by modularity maximization."),
    Doc("d19", "Anthropic released Contextual Retrieval in September 2024, reducing failure 49%."),
    Doc("d20", "Multi-Query Retrieval rewrites one query into many and fuses results via RRF."),
]


SAMPLE_QUERIES: list[tuple[str, str]] = [
    ("Who founded Anthropic?", "Anthropic"),
    ("What is GraphRAG?", "entity graph community"),
    ("How does ReAct loop work?", "thought action observation"),
    ("What is MCP?", "Model Context Protocol"),
    ("How does BM25 score documents?", "sparse retrieval"),
    ("What is hybrid retrieval?", "BM25 dense RRF"),
    ("How does HyDE work?", "hypothetical answer"),
    ("What is ColBERT MaxSim?", "per-token"),
    ("How is RAGAS used?", "faithfulness relevancy"),
    ("What is HippoRAG?", "PageRank entity graph"),
]


def _self_test() -> None:
    v = hash_embed("hello world", dim=16)
    assert len(v) == 16
    assert abs(sum(x * x for x in v) - 1.0) < 1e-6, sum(x * x for x in v)

    assert cosine([1, 0], [1, 0]) > 0.99
    assert cosine([1, 0], [0, 1]) < 0.01

    a = hash_embed("Claude is made by Anthropic")
    b = hash_embed("Anthropic created Claude")
    c = hash_embed("BM25 sparse retrieval scoring")
    assert cosine(a, b) > cosine(a, c), "similar texts should be more similar"

    assert tokenize("Hello, World 123!") == ["hello", "world", "123"]
    assert len(SAMPLE_DOCS) == 20
    assert len(SAMPLE_QUERIES) == 10
    print("[OK] common._self_test passed")


if __name__ == "__main__":
    _self_test()
