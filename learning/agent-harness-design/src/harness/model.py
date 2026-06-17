"""The model boundary — a deterministic MockModel speaking the tool-use protocol.

A real harness calls an LLM that returns either text (end_turn) or tool-use
blocks. We mock that with a `brain(messages) -> ModelResponse` so the *loop*
logic is exercised without any network or weights. The brain inspects the
conversation (esp. prior tool results) to decide the next action — exactly the
contract a real model fulfils.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .util import new_id, est_tokens


@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("call"))

    def signature(self) -> str:
        return f"{self.name}({sorted((self.args or {}).items())})"


@dataclass
class ModelResponse:
    text: str = ""
    tool_calls: list = field(default_factory=list)

    @property
    def stop_reason(self) -> str:
        return "tool_use" if self.tool_calls else "end_turn"

    def out_tokens(self) -> int:
        return est_tokens(self.text) + sum(est_tokens(tc.args) for tc in self.tool_calls)


class MockModel:
    """Wraps a deterministic brain; counts invocations."""

    def __init__(self, brain, name: str = "mock-model"):
        self.brain = brain
        self.name = name
        self.calls = 0

    def respond(self, messages) -> ModelResponse:
        self.calls += 1
        resp = self.brain(messages)
        if not isinstance(resp, ModelResponse):
            raise TypeError("brain must return a ModelResponse")
        return resp


def tool_results_in(messages) -> dict:
    """Helper for brains: map tool-name -> last result value seen so far."""
    out = {}
    for m in messages:
        if m.get("role") == "tool" and "name" in m:
            res = m["content"]
            out[m["name"]] = res.get("value") if isinstance(res, dict) else res
    return out


def _self_test() -> None:
    # A brain that calls one tool, then finishes once it sees the result.
    def brain(messages):
        done = tool_results_in(messages)
        if "ping" not in done:
            return ModelResponse(text="let me ping", tool_calls=[ToolCall("ping")])
        return ModelResponse(text=f"pong was {done['ping']}")

    m = MockModel(brain)
    r1 = m.respond([{"role": "user", "content": "hi"}])
    assert r1.stop_reason == "tool_use" and r1.tool_calls[0].name == "ping"
    assert r1.tool_calls[0].id.startswith("call_")

    r2 = m.respond([{"role": "user", "content": "hi"},
                    {"role": "tool", "name": "ping", "content": {"ok": True, "value": 42}}])
    assert r2.stop_reason == "end_turn" and "42" in r2.text
    assert m.calls == 2 and r1.out_tokens() > 0
    print("[OK] harness.model._self_test passed")


if __name__ == "__main__":
    _self_test()
