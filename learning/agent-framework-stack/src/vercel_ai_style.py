"""Vercel AI SDK-style - generateText/streamText with multi-step tool loop."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Iterator
from common import mock_search


@dataclass
class ToolSpec:
    name: str
    description: str
    execute: Callable[[dict], object]


@dataclass
class StepResult:
    step: int
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)


@dataclass
class GenerateTextResult:
    text: str
    steps: list[StepResult] = field(default_factory=list)
    n_tool_calls: int = 0


def generate_text(
    model_fn: Callable[[str, list], dict],
    prompt: str,
    tools: dict[str, ToolSpec] | None = None,
    max_steps: int = 5,
) -> GenerateTextResult:
    tools = tools or {}
    history: list[dict] = [{"role": "user", "content": prompt}]
    result = GenerateTextResult(text="")

    for step in range(1, max_steps + 1):
        out = model_fn(prompt, history)
        step_result = StepResult(step=step, text=out.get("text", ""))
        history.append({"role": "assistant", "content": step_result.text})

        for tc in out.get("tool_calls", []):
            tool = tools.get(tc["name"])
            if tool is None:
                tool_out = {"error": f"unknown tool: {tc['name']}"}
            else:
                try:
                    tool_out = {"result": tool.execute(tc.get("args", {}))}
                except Exception as e:
                    tool_out = {"error": str(e)}
            step_result.tool_calls.append(tc)
            step_result.tool_results.append(tool_out)
            history.append({"role": "tool", "name": tc["name"], "content": str(tool_out)})

        result.steps.append(step_result)
        result.n_tool_calls += len(step_result.tool_calls)
        if not step_result.tool_calls:
            result.text = step_result.text
            break

    return result


def stream_text(
    model_fn: Callable[[str, list], Iterator[dict]],
    prompt: str,
) -> Iterator[dict]:
    history = [{"role": "user", "content": prompt}]
    for chunk in model_fn(prompt, history):
        yield chunk


def _self_test() -> None:
    search_tool = ToolSpec(
        name="search",
        description="Search KB",
        execute=lambda args: mock_search(args.get("query", ""), k=2),
    )

    def model_fn(prompt: str, history: list) -> dict:
        n_assistant = sum(1 for m in history if m["role"] == "assistant")
        if n_assistant == 0:
            return {"text": "I'll search", "tool_calls": [{"name": "search", "args": {"query": "ReAct"}}]}
        last_tool = next((m for m in reversed(history) if m["role"] == "tool"), None)
        return {"text": f"Based on tool result: {last_tool['content'][:80]}", "tool_calls": []}

    result = generate_text(model_fn, "Search ReAct", tools={"search": search_tool}, max_steps=5)
    assert result.n_tool_calls == 1
    assert "Based on tool result" in result.text
    assert "ReAct" in result.text
    assert len(result.steps) == 2

    def stream_fn(prompt, hist):
        for word in ["hello", " ", "world"]:
            yield {"chunk": word}
    chunks = list(stream_text(stream_fn, "X"))
    assert len(chunks) == 3
    assert chunks[0]["chunk"] == "hello"
    print("[OK] vercel_ai_style._self_test passed")


if __name__ == "__main__":
    _self_test()
