"""Observability — trace spans + cost tracking.

An agent's trajectory is emergent, so without a trace you cannot debug it.
Every harness step emits a span; every model/tool call updates the tracker.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Span:
    kind: str   # system | model | tool | perm | ctx | mem | subagent | loop | done
    label: str
    detail: str = ""


class Trace:
    def __init__(self) -> None:
        self.spans: list[Span] = []

    def add(self, kind: str, label: str, detail: str = "") -> None:
        self.spans.append(Span(kind, label, detail))

    def kinds(self) -> list[str]:
        return [s.kind for s in self.spans]

    def count(self, kind: str) -> int:
        return sum(1 for s in self.spans if s.kind == kind)

    def render(self) -> str:
        return "\n".join(
            f"  [{s.kind:>8}] {s.label}" + (f" :: {s.detail}" if s.detail else "")
            for s in self.spans
        )


@dataclass
class CostTracker:
    model_calls: int = 0
    tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    def add_model(self, tokens_in: int, tokens_out: int) -> None:
        self.model_calls += 1
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out

    def add_tool(self) -> None:
        self.tool_calls += 1

    def usd(self, in_rate: float = 3.0, out_rate: float = 15.0) -> float:
        return round(self.tokens_in / 1e6 * in_rate + self.tokens_out / 1e6 * out_rate, 6)

    def summary(self) -> dict:
        return {
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "usd": self.usd(),
        }


def _self_test() -> None:
    tr = Trace()
    tr.add("model", "turn-0", "wants tool")
    tr.add("tool", "read_config", "ok")
    tr.add("tool", "multiply", "80")
    assert tr.count("tool") == 2 and tr.kinds()[0] == "model"
    assert "read_config" in tr.render()

    ct = CostTracker()
    ct.add_model(100, 20)
    ct.add_tool()
    ct.add_tool()
    s = ct.summary()
    assert s["model_calls"] == 1 and s["tool_calls"] == 2 and s["usd"] > 0
    print("[OK] harness.tracing._self_test passed")


if __name__ == "__main__":
    _self_test()
