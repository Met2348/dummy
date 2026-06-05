"""Capstone — 5 RAG strategies × 4 RAGAS metrics."""
from __future__ import annotations
from common import Doc, SAMPLE_DOCS, SAMPLE_QUERIES
from naive_rag import NaiveRAG
from hybrid import HybridRAG
from reranker_mock import rerank
from hyde_demo import HyDERAG
from graph_rag import GraphRAGMock
from hipporag import HippoRAG
from ragas_metrics import ragas_eval


def make_answer(contexts: list[str], query: str) -> str:
    """Mock answer = concat first 2 contexts as if LLM synthesized."""
    if not contexts:
        return "I have no answer."
    parts = [c[:120] for c in contexts[:2]]
    return " ".join(parts)


def run_naive(docs: list[Doc], queries: list[tuple[str, str]]) -> dict:
    rag = NaiveRAG()
    rag.index(docs)
    return _evaluate(rag.search, queries, label="naive")


def run_hybrid(docs: list[Doc], queries: list[tuple[str, str]]) -> dict:
    rag = HybridRAG()
    rag.index(docs)
    return _evaluate(rag.search, queries, label="hybrid")


def run_hybrid_rerank(docs: list[Doc], queries: list[tuple[str, str]]) -> dict:
    rag = HybridRAG()
    rag.index(docs)

    def search(query: str, k: int = 3):
        results = rag.search(query, k=10)
        return rerank(query, results, k=k)

    return _evaluate(search, queries, label="hybrid+rerank")


def run_hyde(docs: list[Doc], queries: list[tuple[str, str]]) -> dict:
    rag = HyDERAG()
    rag.index(docs)
    return _evaluate(rag.search, queries, label="HyDE")


def run_graphrag(docs: list[Doc], queries: list[tuple[str, str]]) -> dict:
    rag = GraphRAGMock()
    rag.index(docs)
    return _evaluate(rag.query_local, queries, label="GraphRAG")


def run_hipporag(docs: list[Doc], queries: list[tuple[str, str]]) -> dict:
    rag = HippoRAG()
    rag.index(docs)
    return _evaluate(rag.search, queries, label="HippoRAG")


def _evaluate(search_fn, queries: list[tuple[str, str]], label: str) -> dict:
    sums = {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_recall": 0.0}
    n = 0
    for q, gt_keyword in queries:
        try:
            results = search_fn(q, k=3)
        except TypeError:
            results = search_fn(q)
        contexts = [r.chunk.text for r in results]
        answer = make_answer(contexts, q)
        gt_answer = f"The answer relates to {gt_keyword}."
        metrics = ragas_eval(q, answer, contexts, gt_answer)
        for key in sums:
            sums[key] += metrics.get(key, 0.0)
        n += 1
    avg = {k: round(v / max(1, n), 3) for k, v in sums.items()}
    avg["mean"] = round(sum(avg.values()) / 4, 3)
    avg["strategy"] = label
    return avg


def run_compare() -> list[dict]:
    rows = [
        run_naive(SAMPLE_DOCS, SAMPLE_QUERIES),
        run_hybrid(SAMPLE_DOCS, SAMPLE_QUERIES),
        run_hybrid_rerank(SAMPLE_DOCS, SAMPLE_QUERIES),
        run_hyde(SAMPLE_DOCS, SAMPLE_QUERIES),
        run_graphrag(SAMPLE_DOCS, SAMPLE_QUERIES),
        run_hipporag(SAMPLE_DOCS, SAMPLE_QUERIES),
    ]
    return rows


def to_md(rows: list[dict]) -> str:
    lines = [
        "# RAG Capstone — 6 strategies × 4 RAGAS metrics\n",
        f"Docs: {len(SAMPLE_DOCS)} | Queries: {len(SAMPLE_QUERIES)}\n",
        "| Strategy | Faithfulness | Answer-Rel | Ctx-Prec | Ctx-Recall | Mean |",
        "|----------|-------------:|-----------:|---------:|-----------:|-----:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['strategy']:<14} | {r['faithfulness']:.3f} | {r['answer_relevancy']:.3f} | "
            f"{r['context_precision']:.3f} | {r['context_recall']:.3f} | **{r['mean']:.3f}** |"
        )
    best = max(rows, key=lambda r: r["mean"])
    lines.append(f"\n## Winner: **{best['strategy']}** (mean={best['mean']:.3f})\n")
    lines.append("## 5 use case selection\n")
    lines.append("| Use case | Recommend |")
    lines.append("|----------|-----------|")
    lines.append("| Quick PoC | naive |")
    lines.append("| Long-tail terminology | hybrid |")
    lines.append("| High faithfulness | hybrid+rerank |")
    lines.append("| Cross-doc relations | GraphRAG |")
    lines.append("| Multi-hop QA | HippoRAG |")
    return "\n".join(lines)


def _self_test() -> None:
    rows = run_compare()
    assert len(rows) == 6, len(rows)
    strategies = {r["strategy"] for r in rows}
    assert {"naive", "hybrid", "hybrid+rerank", "HyDE", "GraphRAG", "HippoRAG"} == strategies, strategies
    for r in rows:
        assert 0.0 <= r["mean"] <= 1.0, r
    # 至少有一个策略 mean > 0.2
    assert any(r["mean"] > 0.2 for r in rows), [r["mean"] for r in rows]
    print(f"[OK] capstone_rag_compare._self_test passed (6 strategies)")


if __name__ == "__main__":
    _self_test()
    print()
    print(to_md(run_compare()))
