"""DRA retriever — per sub-question, search + fetch."""
from __future__ import annotations
from dra.common import Note, DocSnippet, DRACost
from dra.tools.dra_tools import search_tool, fetch_tool


def retrieve_for_subq(sub_q: str, cost: DRACost, top_k: int = 3) -> Note:
    cost.tool_calls += 1
    hits = search_tool({"query": sub_q})

    findings: list[DocSnippet] = []
    for hit in hits[:top_k]:
        cost.tool_calls += 1
        fetched = fetch_tool({"doc_id": hit["doc_id"]})
        if "error" in fetched:
            continue
        findings.append(DocSnippet(
            doc_id=hit["doc_id"],
            text=fetched["full_text"],
            url=fetched.get("url", ""),
        ))
    return Note(sub_question=sub_q, findings=findings)


def retrieve_all(sub_questions: list[str], cost: DRACost, top_k: int = 3) -> list[Note]:
    return [retrieve_for_subq(q, cost, top_k=top_k) for q in sub_questions]


def _self_test() -> None:
    cost = DRACost()
    note = retrieve_for_subq("vllm paged attention", cost)
    assert len(note.findings) >= 1
    assert note.findings[0].doc_id == "vllm"
    assert cost.tool_calls >= 2

    notes = retrieve_all([
        "PagedAttention vllm",
        "speculative decoding",
        "quantization",
    ], cost)
    assert len(notes) == 3
    assert all(len(n.findings) >= 1 for n in notes), [len(n.findings) for n in notes]
    print("[OK] dra.retriever._self_test passed")


if __name__ == "__main__":
    _self_test()
