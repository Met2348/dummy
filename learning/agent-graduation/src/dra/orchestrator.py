"""DRA orchestrator - Capstone-1 entry point."""
from __future__ import annotations
from dataclasses import dataclass, field
from dra.common import DRACost, Plan, Note, Draft, Verified
from dra.planner import plan_query
from dra.retriever import retrieve_all
from dra.writer import write_report
from dra.verifier import verify_draft
from dra.tools.dra_tools import file_write_tool


CAPSTONE_QUERY = "Write a brief report on modern LLM inference optimization techniques."


@dataclass
class DRARun:
    query: str
    plan: Plan = field(default_factory=Plan)
    notes: list[Note] = field(default_factory=list)
    draft: Draft | None = None
    verified: Verified | None = None
    cost: DRACost = field(default_factory=DRACost)
    output_path: str = "dra_report.md"


def run_dra(query: str = CAPSTONE_QUERY) -> DRARun:
    run = DRARun(query=query)
    run.plan = plan_query(query)
    run.cost.llm_calls += 1
    run.notes = retrieve_all(run.plan.sub_questions, run.cost)
    run.draft = write_report(query, run.notes, run.cost)
    run.verified = verify_draft(run.draft, run.notes, run.cost)
    run.cost.tool_calls += 1
    file_write_tool({"path": run.output_path, "content": run.verified.final_markdown})
    return run


run_capstone_1 = run_dra


def to_md(run: DRARun) -> str:
    lines = [
        "# Deep Research Agent - Capstone-1\n",
        f"**Query:** {run.query}\n",
        "## Plan",
        f"- {len(run.plan.sub_questions)} sub-questions",
        f"- Rationale: {run.plan.rationale}\n",
        "### Sub-questions",
    ]
    for i, q in enumerate(run.plan.sub_questions, start=1):
        lines.append(f"{i}. {q}")

    n_findings = sum(len(n.findings) for n in run.notes)
    lines.append(f"\n## Retrieval: {n_findings} doc snippets across {len(run.notes)} sub-q")

    lines.append("\n## Draft preview (first 400 chars)")
    if run.draft:
        lines.append("```")
        lines.append(run.draft.markdown[:400])
        lines.append("```")
        lines.append(f"\n- Citations: {len(run.draft.citations)}")
        lines.append(f"- Claims: {len(run.draft.claims)}")

    if run.verified:
        sup = len(run.verified.supported_claims)
        unsup = len(run.verified.unsupported_claims)
        lines.append(f"\n## Verification")
        lines.append(f"- Supported: {sup}")
        lines.append(f"- Unsupported: {unsup}")

    lines.append("\n## Cost")
    lines.append(f"- LLM calls: {run.cost.llm_calls}")
    lines.append(f"- Tool calls: {run.cost.tool_calls}")
    lines.append(f"- Tokens in: {run.cost.tokens_in}")
    lines.append(f"- Tokens out: {run.cost.tokens_out}")
    lines.append(f"- ~cost_usd: {run.cost.usd()}")

    n_subq = len(run.plan.sub_questions)
    n_cites = len(run.draft.citations) if run.draft else 0
    n_unsup = len(run.verified.unsupported_claims) if run.verified else 999
    passed = n_subq == 5 and n_cites >= 3 and n_unsup == 0
    lines.append(f"\n## Verdict: {'[PASS]' if passed else '[FAIL]'}")
    return "\n".join(lines)


def _self_test() -> None:
    run = run_capstone_1()
    assert len(run.plan.sub_questions) == 5
    assert sum(len(n.findings) for n in run.notes) >= 5
    assert run.draft is not None
    assert len(run.draft.citations) >= 3, len(run.draft.citations)
    assert run.verified is not None
    assert len(run.verified.supported_claims) >= 5, len(run.verified.supported_claims)
    assert run.cost.llm_calls >= 3
    assert run.cost.tool_calls >= 5
    from dra.tools.dra_tools import get_fs
    assert "dra_report.md" in get_fs()
    print(f"[OK] dra.orchestrator._self_test passed ({len(run.draft.citations)} cites, {len(run.verified.supported_claims)} supported)")


if __name__ == "__main__":
    _self_test()
    print()
    print(to_md(run_capstone_1()))
