"""Common dataclasses & helpers for agent-foundations.

Stdlib-only. Mock LLM via keyword-pattern matching.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import re


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    func: Callable[[dict], Any]


@dataclass
class ActionResult:
    ok: bool
    value: Any = None
    error: Optional[str] = None

    def to_obs(self) -> str:
        if not self.ok:
            return f"ERROR: {self.error}"
        return str(self.value)


@dataclass
class Step:
    step_num: int
    thought: str
    action: str  # "tool_name(args)" or "FINAL"
    args: dict = field(default_factory=dict)
    observation: str = ""


@dataclass
class Trace:
    question: str = ""
    steps: list[Step] = field(default_factory=list)
    final: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0

    def add(self, step: Step) -> None:
        self.steps.append(step)

    def to_md(self) -> str:
        lines = [f"# Trace\n\nQ: {self.question}\n", "| # | Thought | Action | Obs |", "|---|---------|--------|-----|"]
        for s in self.steps:
            lines.append(
                f"| {s.step_num} | {s.thought[:40]} | {s.action} | {s.observation[:40]} |"
            )
        lines.append(f"\nFinal: {self.final}")
        return "\n".join(lines)


def parse_action(text: str) -> tuple[str, dict]:
    """Parse a single 'Action: tool_name(args)' line.

    Supports `tool(key=value, k2=value2)` and `tool("string")`.
    """
    m = re.search(r"Action\s*\d*:\s*(\w+)\s*\((.*?)\)\s*(?:\n|$)", text)
    if not m:
        return "", {}
    name = m.group(1)
    raw = m.group(2).strip()
    args: dict[str, Any] = {}
    if not raw:
        return name, args
    if "=" in raw:
        for part in re.split(r",\s*(?=\w+=)", raw):
            if "=" in part:
                k, v = part.split("=", 1)
                args[k.strip()] = _coerce(v.strip())
    else:
        args["input"] = _coerce(raw)
    return name, args


def parse_final(text: str) -> Optional[str]:
    m = re.search(r"Final\s*Answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def _coerce(s: str) -> Any:
    s = s.strip()
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    if s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def make_pattern_llm(rules: list[tuple[str, str]]) -> Callable[[str], str]:
    """Mock LLM: scan prompt for keyword, output canned reply.

    Returns first matching rule. Use last rule as default.
    """
    compiled = [(re.compile(p, re.IGNORECASE), reply) for p, reply in rules]

    def llm(prompt: str) -> str:
        for pat, reply in compiled:
            if pat.search(prompt):
                return reply
        return "Thought: I don't know what to do.\nFinal Answer: unsure"

    return llm


def _self_test() -> None:
    name, args = parse_action("Thought: ...\nAction: search(query=foo)\n")
    assert name == "search" and args == {"input" if "input" in args else "query": "foo"} or args.get("query") == "foo"

    name, args = parse_action('Action: calc("2+3")')
    assert name == "calc" and args.get("input") == "2+3", args

    assert parse_final("Final Answer: 42") == "42"
    assert parse_final("Final Answer:   hello world\n") == "hello world"

    llm = make_pattern_llm([(r"compute", "Action: calc(2+3)"), (r".*", "Final Answer: idk")])
    assert "calc" in llm("Please compute")
    assert "idk" in llm("Random")

    print("[OK] common._self_test passed")


if __name__ == "__main__":
    _self_test()
