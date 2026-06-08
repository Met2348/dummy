"""Capstone - same search+summary task in 3 framework styles."""
from __future__ import annotations
from dataclasses import dataclass, field
from common import mock_search, mock_summarize, FrameworkRun
from langchain_style import PromptTemplate, MockLLM, RunnableLambda
from vercel_ai_style import generate_text, ToolSpec
from claude_agent_sdk_style import query


CAPSTONE_QUERY = "What is ReAct agent pattern?"


def run_langchain_style() -> FrameworkRun:
    def retrieve(q: str) -> str:
        return " | ".join(mock_search(q, k=3))

    def synth(d: dict) -> str:
        return mock_summarize(d["q"], d["contexts"].split(" | "))

    retrieve_step = RunnableLambda(retrieve)
    chain = (
        RunnableLambda(lambda q: {"q": q, "contexts": retrieve(q)})
        | RunnableLambda(synth)
    )
    output = chain.invoke(CAPSTONE_QUERY)
    return FrameworkRun(framework="LangChain LCEL", loc=8, output=output,
                        abstraction_level="Runnable chain via pipe")


def run_vercel_style() -> FrameworkRun:
    search_tool = ToolSpec(
        name="search", description="search KB",
        execute=lambda args: mock_search(args.get("query", ""), k=3),
    )

    def model_fn(prompt: str, history: list) -> dict:
        n_assistant = sum(1 for m in history if m["role"] == "assistant")
        if n_assistant == 0:
            return {"text": "Will search.", "tool_calls": [{"name": "search", "args": {"query": prompt}}]}
        last_tool = next((m for m in reversed(history) if m["role"] == "tool"), None)
        return {"text": mock_summarize(prompt, [last_tool["content"]]), "tool_calls": []}

    result = generate_text(model_fn, CAPSTONE_QUERY, tools={"search": search_tool})
    return FrameworkRun(framework="Vercel AI SDK", loc=10, output=result.text,
                        abstraction_level="generateText + tool")


def run_claude_sdk_style() -> FrameworkRun:
    def mock_llm(prompt: str, history: list):
        return [
            {"tool": "WebSearch", "args": {"query": prompt}},
            {"text": mock_summarize(prompt, [str(BUILTIN_RESULT)])},
        ]
    res = query(CAPSTONE_QUERY,
                options={"allowedTools": ["WebSearch"]},
                mock_llm=mock_llm)
    final = next((m["content"] for m in res.messages
                  if m["role"] == "assistant" and m["content"].startswith("Summary")), "")
    return FrameworkRun(framework="Claude Agent SDK", loc=3, output=final,
                        abstraction_level="query() one-liner")


BUILTIN_RESULT = mock_search("ReAct", k=3)


def run_capstone() -> list[FrameworkRun]:
    return [
        run_langchain_style(),
        run_vercel_style(),
        run_claude_sdk_style(),
    ]


def to_md(runs: list[FrameworkRun]) -> str:
    lines = [
        "# Same-Task 3-Framework Capstone\n",
        f"Task: {CAPSTONE_QUERY}\n",
        "| Framework | LoC | Abstraction | Output (truncated) |",
        "|-----------|----:|-------------|---------------------|",
    ]
    for r in runs:
        out = r.output.replace("\n", " ")[:60]
        lines.append(f"| {r.framework} | {r.loc} | {r.abstraction_level} | {out} |")
    all_react = all("ReAct" in r.output for r in runs)
    verdict = "[PASS]" if all_react else "[FAIL]"
    lines.append(f"\n## Verdict: {verdict} (all 3 frameworks produced ReAct-related summary)")
    return "\n".join(lines)


def _self_test() -> None:
    runs = run_capstone()
    assert len(runs) == 3
    fws = {r.framework for r in runs}
    assert {"LangChain LCEL", "Vercel AI SDK", "Claude Agent SDK"} == fws

    for r in runs:
        assert "ReAct" in r.output, (r.framework, r.output)
        assert r.loc > 0

    locs = {r.framework: r.loc for r in runs}
    assert locs["Claude Agent SDK"] < locs["LangChain LCEL"], locs
    print("[OK] capstone_same_task._self_test passed (3 frameworks, all produced ReAct summary)")


if __name__ == "__main__":
    _self_test()
    print()
    print(to_md(run_capstone()))
