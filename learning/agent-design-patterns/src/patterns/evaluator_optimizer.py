"""Workflow pattern 5 — Evaluator-Optimizer.

A generator produces a draft; an evaluator scores it and returns concrete
feedback; the generator revises. Loop until the evaluator passes or a budget is
hit. Use when you have clear evaluation criteria AND iteration measurably helps
(translation, code that must pass tests, structured extraction).

    task -> [generate] -> [evaluate] --pass--> output
              ^               |
              +---feedback----+ (fail)
"""
from __future__ import annotations

from common import MockLLM, CostTracker, Trace, PatternResult, preview


def run_eval_opt(task, generator, evaluator, max_iters=4, tracker=None):
    """generator: fn(task, feedback)->str; evaluator: fn(str)->(ok, feedback)."""
    trace = Trace()
    feedback, output = "", ""
    for i in range(max_iters):
        output = generator(task, feedback)
        trace.add("llm", f"generate#{i}", preview(output))
        ok, fb = evaluator(output)
        trace.add("eval", f"evaluate#{i}", "PASS" if ok else fb)
        if ok:
            return PatternResult("evaluator_optimizer", output, trace, tracker, ok=True)
        feedback = fb
    return PatternResult("evaluator_optimizer", output, trace, tracker, ok=False)


def demo():
    tracker = CostTracker()
    required = ["title", "priority", "labels"]
    values = {"title": "title:Fix login", "priority": "priority:high", "labels": "labels:[auth,bug]"}
    state = {"have": {}}  # closure state (no mutable default arg)

    def generator(task, feedback):
        # Mock "improvement": add exactly one still-missing field per round,
        # carrying forward what it already has -> visible iterative progress.
        for f in required:
            if f not in state["have"]:
                state["have"][f] = values[f]
                break
        text = " ".join(state["have"][f] for f in required if f in state["have"])
        return MockLLM(responder=lambda p, **k: text, tracker=tracker).complete(task)

    def evaluator(output):
        missing = [f for f in required if f not in output]
        if not missing:
            return True, ""
        return False, "missing: " + ",".join(missing)

    return run_eval_opt("write a bug ticket", generator, evaluator, max_iters=5, tracker=tracker)


def _self_test() -> None:
    r = demo()
    assert r.ok, "should converge once all fields present"
    # 1st draft has only title -> needs 3 generate rounds (title, +priority, +labels).
    gen_steps = [k for k in r.trace.kinds() if k == "llm"]
    assert len(gen_steps) == 3, len(gen_steps)
    assert all(f in r.output for f in ("title", "priority", "labels"))

    # Budget exhaustion returns ok=False.
    tracker = CostTracker()
    never = run_eval_opt(
        "x",
        generator=lambda t, fb: "always-bad",
        evaluator=lambda o: (False, "nope"),
        max_iters=2,
        tracker=tracker,
    )
    assert not never.ok
    print("[OK] evaluator_optimizer._self_test passed")


if __name__ == "__main__":
    _self_test()
    print(demo().trace.render())
