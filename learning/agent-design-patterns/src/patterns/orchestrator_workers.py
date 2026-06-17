"""Workflow pattern 4 — Orchestrator-Workers.

Like sectioning, but the subtasks are NOT known up front: an orchestrator LLM
*dynamically* decomposes the input, spins up a worker per subtask, then
synthesizes. Use when you can't predict how many/which subtasks a task needs
(e.g. "edit every file that imports X" — count depends on the codebase).

    input -> [orchestrator: plan subtasks] -> [worker]* -> [synthesize] -> output
"""
from __future__ import annotations

from common import MockLLM, CostTracker, Trace, PatternResult, preview


def run_orchestrator(text, planner, worker, synthesize, tracker=None):
    """planner: fn(str)->list[str]; worker: fn(str)->str; synthesize: fn(str, list)->any."""
    trace = Trace()
    subtasks = planner(text)
    trace.add("plan", "decompose", f"{len(subtasks)} subtasks: {subtasks}")
    results = []
    for st in subtasks:
        r = worker(st)
        results.append((st, r))
        trace.add("llm", f"worker:{st}", preview(r))
    final = synthesize(text, results)
    trace.add("done", "synthesize", preview(final))
    return PatternResult("orchestrator_workers", final, trace, tracker, ok=len(subtasks) > 0)


def demo():
    tracker = CostTracker()

    def planner(request, **k):
        # Subtasks depend on what the request actually asks about — dynamic.
        aspects = []
        r = request.lower()
        for kw, name in (("pric", "pricing"), ("compet", "competitors"),
                         ("risk", "risks"), ("market", "market-size")):
            if kw in r:
                aspects.append(name)
        return aspects or ["overview"]

    def worker(subtask):
        findings = {
            "pricing": "freemium + $20/mo pro tier",
            "competitors": "3 incumbents, 1 fast-growing startup",
            "risks": "regulatory uncertainty in EU",
            "market-size": "$4B TAM growing 18%/yr",
            "overview": "general summary",
        }
        return MockLLM(
            responder=lambda p, **k: findings.get(subtask, "n/a"), tracker=tracker
        ).complete(f"research {subtask}")

    def synthesize(request, results):
        return "REPORT: " + "; ".join(f"{k}={v}" for k, v in results)

    return run_orchestrator(
        "Research the pricing, competitors and market for product Z",
        planner, worker, synthesize, tracker=tracker,
    )


def _self_test() -> None:
    r = demo()
    # Planner detected exactly the 3 mentioned aspects (not the 4th, "risks").
    assert r.ok and r.tracker.calls == 3, r.tracker.calls
    assert "pricing" in r.output and "market-size" in r.output
    assert "risks" not in r.output, "planner should not invent unrequested subtasks"

    # Dynamic decomposition: subtask count is data-driven, not fixed.
    tracker = CostTracker()
    r2 = run_orchestrator(
        "split a, b",
        planner=lambda t: [p.strip() for p in t.replace("split", "").split(",")],
        worker=lambda st: f"did {st}",
        synthesize=lambda t, rs: [k for k, _ in rs],
        tracker=tracker,
    )
    assert r2.output == ["a", "b"] and tracker.calls == 0  # worker here is pure, no LLM
    print("[OK] orchestrator_workers._self_test passed")


if __name__ == "__main__":
    _self_test()
    print(demo().trace.render())
