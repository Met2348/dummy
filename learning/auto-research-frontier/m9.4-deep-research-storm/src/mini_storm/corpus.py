"""一个带"事实账本"的小型文献库——每篇文档不仅有标题，还显式记录它**真正支持哪些论断**。

这是 9.4 的关键设计：要核查"引用忠实度"，就必须知道每篇文献到底说了什么。
真实系统里这一步靠 NLI/蕴含模型；这里用显式 `supports` 集合 + 集合包含，纯 CPU、确定性、可测。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    keywords: tuple        # 用于检索
    supports: frozenset    # 它**真正支持**的论断 token（事实账本）


CORPUS = (
    Doc("2504.08066", "The AI Scientist v2",
        ("tree-search", "experiment", "autonomous", "agent"),
        frozenset({"tree-search", "experiment-manager", "lower-success-when-more-autonomous",
                   "vlm-feedback"})),
    Doc("2502.18864", "Towards an AI co-scientist",
        ("hypothesis", "multi-agent", "biomedical"),
        frozenset({"hypothesis-generation", "wet-lab-validated", "multi-agent-debate"})),
    Doc("2402.14207", "Assisting Writing with STORM",
        ("retrieval", "perspective", "outline", "wikipedia"),
        frozenset({"multi-perspective", "outline-driven", "retrieval-augmented",
                   "wikipedia-style"})),
    Doc("2506.20803", "The Ideation-Execution Gap",
        ("ideation", "execution", "feasibility"),
        frozenset({"novel-not-feasible", "execution-reveals", "human-beats-ai-on-execution"})),
    Doc("2509.08713", "Hidden Pitfalls of AI Research Agents",
        ("pitfalls", "hallucination", "integrity"),
        frozenset({"hallucinated-results", "benchmark-gaming", "dataset-swap",
                   "self-review-inflates"})),
    Doc("2411.14199", "OpenScholar",
        ("retrieval", "citation", "datastore", "scholar"),
        frozenset({"datastore-45M", "citation-accuracy", "retrieval-corpus"})),
)

BY_ID = {d.doc_id: d for d in CORPUS}


def _tokens(text: str) -> set:
    return {t.strip(".,?!").lower() for t in text.replace("-", " ").split() if t}


def retrieve(query: str, k: int = 3) -> list:
    """关键词重叠检索，确定性（平手按 doc_id）。"""
    q = _tokens(query)
    scored = []
    for d in CORPUS:
        hay = _tokens(d.title) | set(d.keywords)
        s = len(q & hay)
        if s:
            scored.append((s, d))
    scored.sort(key=lambda sd: (-sd[0], sd[1].doc_id))
    return [d for _, d in scored[:k]]
