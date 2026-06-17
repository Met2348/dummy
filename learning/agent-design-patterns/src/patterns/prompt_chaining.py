"""Workflow pattern 1 — Prompt Chaining.

Decompose a task into a FIXED sequence of LLM calls, each consuming the
previous output. Optional programmatic *gates* between steps let you fail fast
or branch. Use when the task cleanly splits into ordered subtasks and you trade
a little latency for much higher per-step accuracy.

    input -> [LLM step 1] -(gate)-> [LLM step 2] -> ... -> output
"""
from __future__ import annotations

from common import MockLLM, CostTracker, Trace, PatternResult, preview


def run_chain(text, steps, gates=None, tracker=None):
    """steps: list[(name, fn(str)->str)]. gates: {step_name: fn(str)->(ok, reason)}."""
    trace = Trace()
    gates = gates or {}
    cur = text
    for name, fn in steps:
        cur = fn(cur)
        trace.add("llm", name, preview(cur))
        if name in gates:
            ok, reason = gates[name](cur)
            trace.add("gate", f"{name}-gate", reason)
            if not ok:
                return PatternResult("prompt_chaining", cur, trace, tracker, ok=False)
    return PatternResult("prompt_chaining", cur, trace, tracker, ok=True)


def demo():
    tracker = CostTracker()

    def brain(prompt, **k):
        if "outline" in prompt:
            return "- intro\n- benefits\n- call to action"
        if "expand" in prompt:
            return "Our widget saves time. It is fast. It is cheap."
        if "punchy" in prompt:
            return "Save HOURS with our blazing-fast widget — try it free!"
        return "[draft]"

    llm = MockLLM(responder=brain, tracker=tracker)
    steps = [
        ("outline", lambda t: llm.complete(f"outline for: {t}")),
        ("expand", lambda t: llm.complete(f"expand outline into copy: {t}")),
        ("polish", lambda t: llm.complete(f"make it punchy: {t}")),
    ]
    gates = {"outline": lambda o: (o.count("-") >= 2, f"{o.count('-')} bullets")}
    return run_chain("a productivity widget", steps, gates, tracker=tracker)


def _self_test() -> None:
    r = demo()
    assert r.ok, "chain should pass the outline gate"
    assert "gate" in r.trace.kinds()
    assert r.tracker.calls == 3, r.tracker.calls
    assert "widget" in r.output.lower()

    # A failing gate must short-circuit (no further LLM calls).
    tracker = CostTracker()
    llm = MockLLM(responder=lambda p, **k: "no bullets here", tracker=tracker)
    bad = run_chain(
        "x",
        [("outline", lambda t: llm.complete(t)), ("expand", lambda t: llm.complete(t))],
        gates={"outline": lambda o: (o.count("-") >= 2, "too few")},
        tracker=tracker,
    )
    assert not bad.ok and tracker.calls == 1, "gate should stop the chain early"
    print("[OK] prompt_chaining._self_test passed")


if __name__ == "__main__":
    _self_test()
    print(demo().trace.render())
