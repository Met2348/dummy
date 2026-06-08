"""DRA writer - synthesize markdown report from notes."""
from __future__ import annotations
from dra.common import Note, Draft, Citation, DRACost


def write_report(query: str, notes: list[Note], cost: DRACost) -> Draft:
    cost.llm_calls += 1
    cost.tokens_in += sum(
        len(n.sub_question) // 4 + sum(len(f.text) // 4 for f in n.findings)
        for n in notes
    )

    lines: list[str] = []
    citations: list[Citation] = []
    claims: list[str] = []

    title = f"# Report: {query}"
    lines.append(title)
    lines.append("")
    lines.append(f"This report covers {len(notes)} aspects, drawing on {sum(len(n.findings) for n in notes)} sources.")
    lines.append("")

    cite_idx = 0
    seen_doc_to_idx: dict[str, int] = {}

    for i, note in enumerate(notes, start=1):
        section_title = f"## {i}. {note.sub_question}"
        lines.append(section_title)
        if not note.findings:
            lines.append("No specific source found.")
            continue
        for finding in note.findings:
            if finding.doc_id in seen_doc_to_idx:
                cite_n = seen_doc_to_idx[finding.doc_id]
            else:
                cite_idx += 1
                cite_n = cite_idx
                seen_doc_to_idx[finding.doc_id] = cite_n
                citations.append(Citation(id=cite_n, text=finding.url, doc_id=finding.doc_id))
            sentence = f"- {finding.text} [{cite_n}]"
            lines.append(sentence)
            claims.append(finding.text)
        lines.append("")

    lines.append("## References")
    for cit in citations:
        lines.append(f"[{cit.id}] {cit.text} (source: {cit.doc_id})")

    markdown = "\n".join(lines)
    cost.tokens_out += len(markdown) // 4
    return Draft(markdown=markdown, claims=claims, citations=citations)


def _self_test() -> None:
    from dra.retriever import retrieve_all

    cost = DRACost()
    notes = retrieve_all([
        "vllm PagedAttention",
        "speculative decoding",
        "GPTQ AWQ quantization",
    ], cost)
    draft = write_report("LLM inference", notes, cost)
    assert "# Report" in draft.markdown
    assert "## References" in draft.markdown
    assert len(draft.citations) >= 2
    assert len(draft.claims) >= 2
    assert "[1]" in draft.markdown
    assert cost.llm_calls >= 1
    print("[OK] dra.writer._self_test passed")


if __name__ == "__main__":
    _self_test()
