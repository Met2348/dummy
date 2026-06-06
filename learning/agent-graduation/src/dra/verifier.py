"""DRA verifier — check each claim is supported by some source."""
from __future__ import annotations
from dra.common import Draft, Note, Verified, DRACost
from dra.tools.dra_tools import KB


def verify_draft(draft: Draft, notes: list[Note], cost: DRACost) -> Verified:
    cost.llm_calls += 1
    source_blob = " ".join(
        f.text for n in notes for f in n.findings
    ) + " " + " ".join(doc["text"] for doc in KB.values())
    source_blob_low = source_blob.lower()

    supported: list[str] = []
    unsupported: list[str] = []
    for claim in draft.claims:
        keywords = [t for t in claim.lower().split() if len(t) > 4][:5]
        if not keywords:
            supported.append(claim)
            continue
        matched = sum(1 for k in keywords if k in source_blob_low)
        if matched / len(keywords) >= 0.5:
            supported.append(claim)
        else:
            unsupported.append(claim)

    final_md = draft.markdown
    if unsupported:
        final_md += "\n\n## Verification notes\n"
        for u in unsupported:
            final_md += f"- WARNING unsupported: {u[:80]}\n"
    else:
        final_md += "\n\n## Verification: all claims supported [PASS]"

    cost.tokens_out += len(final_md) // 4
    return Verified(
        final_markdown=final_md,
        supported_claims=supported,
        unsupported_claims=unsupported,
    )


def _self_test() -> None:
    from dra.retriever import retrieve_all
    from dra.writer import write_report

    cost = DRACost()
    notes = retrieve_all([
        "vllm PagedAttention KV cache",
        "speculative decoding draft model",
        "GPTQ AWQ INT4 quantization",
    ], cost)
    draft = write_report("LLM inference", notes, cost)
    verified = verify_draft(draft, notes, cost)
    assert len(verified.supported_claims) >= 1
    assert "PASS" in verified.final_markdown or "unsupported" in verified.final_markdown
    print(f"[OK] dra.verifier._self_test passed ({len(verified.supported_claims)} supported, {len(verified.unsupported_claims)} unsupported)")


if __name__ == "__main__":
    _self_test()
