"""Workflow pattern 3 — Parallelization.

Two flavors, both run independent LLM calls concurrently (here: sequentially,
but logically independent) and aggregate:

  * Sectioning — split a task into independent subtasks, one call each, combine.
  * Voting     — run the SAME task N times with variation, take a consensus.

Use sectioning for speed/separation of concerns; use voting for reliability on
a single hard judgement.
"""
from __future__ import annotations

from collections import Counter

from common import MockLLM, CostTracker, Trace, PatternResult, preview


def run_sectioning(subtasks, worker, aggregate, tracker=None):
    """subtasks: list[str]; worker: fn(str)->str; aggregate: fn(list)->any."""
    trace = Trace()
    results = []
    for st in subtasks:
        r = worker(st)
        results.append(r)
        trace.add("section", st, preview(r))
    final = aggregate(results)
    trace.add("done", "aggregate", preview(final))
    return PatternResult("parallelization:sectioning", final, trace, tracker, ok=True)


def run_voting(text, worker, n, aggregate, tracker=None):
    """worker: fn(text, variant:int)->str; aggregate: fn(list)->any."""
    trace = Trace()
    votes = [worker(text, variant=i) for i in range(n)]
    for i, v in enumerate(votes):
        trace.add("vote", f"run{i}", preview(v))
    final = aggregate(votes)
    trace.add("done", "consensus", preview(final))
    ok = votes.count(final) > n // 2  # majority actually agreed
    return PatternResult("parallelization:voting", final, trace, tracker, ok=ok)


def demo_sectioning():
    tracker = CostTracker()
    scores = {"price": "8/10 affordable", "quality": "9/10 solid", "support": "7/10 ok"}

    def worker(aspect):
        return MockLLM(
            responder=lambda p, **k: scores[aspect], tracker=tracker
        ).complete(f"rate {aspect}")

    return run_sectioning(
        ["price", "quality", "support"],
        worker,
        aggregate=lambda rs: " | ".join(rs),
        tracker=tracker,
    )


def demo_voting():
    tracker = CostTracker()
    # Three of five framings say "positive"; voting should overrule the outliers.
    answers = ["positive", "positive", "neutral", "positive", "negative"]

    def worker(text, variant):
        return MockLLM(
            responder=lambda p, **k: answers[variant], tracker=tracker
        ).complete(f"[variant {variant}] sentiment of: {text}")

    return run_voting(
        "the product is great but shipping was slow",
        worker,
        n=5,
        aggregate=lambda vs: Counter(vs).most_common(1)[0][0],
        tracker=tracker,
    )


def _self_test() -> None:
    s = demo_sectioning()
    assert s.ok and s.tracker.calls == 3
    assert s.trace.kinds().count("section") == 3
    assert "price" not in s.output and "affordable" in s.output

    v = demo_voting()
    assert v.output == "positive", v.output
    assert v.ok and v.tracker.calls == 5
    print("[OK] parallelization._self_test passed")


if __name__ == "__main__":
    _self_test()
    print("-- sectioning --")
    print(demo_sectioning().trace.render())
    print("-- voting --")
    print(demo_voting().trace.render())
