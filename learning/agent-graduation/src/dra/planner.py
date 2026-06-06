"""DRA planner — decompose query to sub-questions."""
from __future__ import annotations
from dra.common import Plan


SUB_QUESTION_TEMPLATES = {
    "llm inference": [
        "What are major LLM inference engines in 2026?",
        "How do PagedAttention and RadixAttention compare?",
        "What is speculative decoding and how much speedup does it give?",
        "What quantization methods are most popular?",
        "How does FlashAttention v3 affect inference?",
    ],
    "rag": [
        "What is RAG architecture?",
        "How does hybrid retrieval improve recall?",
        "What is GraphRAG?",
        "How do rerankers improve precision?",
        "What is RAGAS evaluation?",
    ],
    "agent": [
        "What is the ReAct pattern?",
        "How does LangGraph differ from LangChain?",
        "What is MCP?",
        "How do multi-agent systems coordinate?",
        "What memory layers do agents need?",
    ],
}


def plan_query(query: str) -> Plan:
    q_low = query.lower()
    for key, templates in SUB_QUESTION_TEMPLATES.items():
        if key in q_low:
            return Plan(
                sub_questions=templates,
                rationale=f"Decomposed into {len(templates)} sub-questions on '{key}'.",
            )
    return Plan(
        sub_questions=[
            f"What is {query}?",
            f"How does {query} work in 2026?",
            f"What are major implementations?",
            f"What are limitations?",
            f"What are 2025-2026 trends?",
        ],
        rationale="Generic 5-step decomposition fallback.",
    )


def _self_test() -> None:
    p = plan_query("Write a brief report on 2026 LLM inference optimization techniques.")
    assert len(p.sub_questions) == 5
    assert "inference" in p.rationale or "Decomposed" in p.rationale
    assert any("Paged" in s or "Speculative" in s or "FlashAttention" in s for s in p.sub_questions)

    p2 = plan_query("Tell me about RAG")
    assert len(p2.sub_questions) == 5
    assert any("RAG" in s for s in p2.sub_questions)

    p3 = plan_query("Quantum computing basics")
    assert len(p3.sub_questions) == 5
    assert "Generic" in p3.rationale
    print("[OK] dra.planner._self_test passed")


if __name__ == "__main__":
    _self_test()
