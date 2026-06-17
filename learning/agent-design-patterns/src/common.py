"""Shared mock infra for agent-design-patterns.

stdlib-only, CPU-only, fully deterministic. The "LLM" here is a thin mock:
the *lesson* of each pattern is the orchestration logic around the LLM, not
the LLM itself, so we keep the model a deterministic responder and make the
control flow (chain / route / parallel / orchestrate / evaluate / loop) the
star.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any


def est_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token, the usual back-of-envelope)."""
    return max(1, len(str(text)) // 4)


def preview(text: Any, n: int = 56) -> str:
    """One-line preview for traces."""
    s = str(text).replace("\n", " / ")
    return s if len(s) <= n else s[: n - 1] + "…"


@dataclass
class CostTracker:
    """Counts LLM calls + tokens so patterns can be compared on cost."""

    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    def add(self, prompt: str, output: str) -> None:
        self.calls += 1
        self.tokens_in += est_tokens(prompt)
        self.tokens_out += est_tokens(output)

    def usd(self, in_rate: float = 3.0, out_rate: float = 15.0) -> float:
        """Mock pricing: $3/M in, $15/M out (Sonnet-ish)."""
        return round(self.tokens_in / 1e6 * in_rate + self.tokens_out / 1e6 * out_rate, 6)

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "usd": self.usd(),
        }


Responder = Callable[..., str]


class MockLLM:
    """Deterministic mock LLM. `responder(prompt, **kw) -> str` is injected by
    each pattern so the orchestration stays the focus. Tracks cost."""

    def __init__(
        self,
        responder: Responder | None = None,
        tracker: CostTracker | None = None,
        name: str = "mock",
    ):
        self.name = name
        self.responder = responder or (lambda p, **k: f"[echo:{str(p)[:30]}]")
        self.tracker = tracker if tracker is not None else CostTracker()

    def complete(self, prompt: str, **kw) -> str:
        out = self.responder(prompt, **kw)
        self.tracker.add(prompt, out)
        return out


@dataclass
class Tool:
    """A callable the agent may invoke."""

    name: str
    description: str
    fn: Callable[..., Any]

    def __call__(self, *a, **k):
        return self.fn(*a, **k)


@dataclass
class TraceStep:
    kind: str  # llm | tool | gate | route | section | vote | plan | eval | loop | done
    label: str
    detail: str = ""


class Trace:
    """Append-only execution trace — the spine of pattern observability."""

    def __init__(self) -> None:
        self.steps: list[TraceStep] = []

    def add(self, kind: str, label: str, detail: str = "") -> None:
        self.steps.append(TraceStep(kind, label, detail))

    def kinds(self) -> list[str]:
        return [s.kind for s in self.steps]

    def render(self) -> str:
        return "\n".join(
            f"  [{s.kind:>7}] {s.label}" + (f" :: {s.detail}" if s.detail else "")
            for s in self.steps
        )


@dataclass
class PatternResult:
    """Uniform return type so the capstone can compare patterns apples-to-apples."""

    pattern: str
    output: Any
    trace: Trace
    tracker: CostTracker
    ok: bool = True

    def row(self) -> dict:
        s = self.tracker.summary()
        return {
            "pattern": self.pattern,
            "calls": s["calls"],
            "tokens": s["tokens_in"] + s["tokens_out"],
            "usd": s["usd"],
            "steps": len(self.trace.steps),
            "ok": self.ok,
        }


def _self_test() -> None:
    assert est_tokens("abcdefgh") == 2
    t = CostTracker()
    llm = MockLLM(responder=lambda p, **k: "hello world", tracker=t)
    out = llm.complete("say hi")
    assert out == "hello world"
    assert t.calls == 1 and t.tokens_in > 0 and t.tokens_out > 0
    assert t.usd() > 0

    tr = Trace()
    tr.add("llm", "draft", "wrote 3 lines")
    tr.add("gate", "length-check", "pass")
    assert tr.kinds() == ["llm", "gate"]
    assert "draft" in tr.render()

    tool = Tool("calc", "adds", lambda a, b: a + b)
    assert tool(2, 3) == 5

    pr = PatternResult("demo", "out", tr, t)
    assert pr.row()["pattern"] == "demo"
    print("[OK] common._self_test passed")


if __name__ == "__main__":
    _self_test()
