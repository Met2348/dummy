"""Workflow pattern 2 — Routing.

Classify the input, then dispatch to a specialized handler. Separating
classification from handling lets each handler be optimized (different prompt,
model, or tools) without one giant do-everything prompt. Use when inputs fall
into distinct categories better served separately.

    input -> [classifier] -> category -> [specialized handler] -> output
"""
from __future__ import annotations

from common import MockLLM, CostTracker, Trace, PatternResult, preview


def run_router(text, classifier, handlers, tracker=None):
    """classifier: fn(str)->category. handlers: {category: fn(str)->str}."""
    trace = Trace()
    category = classifier(text)
    trace.add("route", "classify", category)
    known = category in handlers
    handler = handlers.get(category) or handlers["default"]
    out = handler(text)
    trace.add("llm", f"handle:{category if known else 'default'}", preview(out))
    return PatternResult("routing", out, trace, tracker, ok=known)


def demo():
    tracker = CostTracker()

    def classify(text, **k):
        t = text.lower()
        if any(w in t for w in ("refund", "charge", "invoice", "bill")):
            return "billing"
        if any(w in t for w in ("error", "crash", "bug", "broken", "login")):
            return "technical"
        return "general"

    def make_handler(kind):
        replies = {
            "billing": "Routed to billing: I can process your refund.",
            "technical": "Routed to tech support: let's debug the error.",
            "general": "Routed to general help: how can I assist?",
        }
        return lambda text: MockLLM(
            responder=lambda p, **k: replies[kind], tracker=tracker
        ).complete(text)

    handlers = {k: make_handler(k) for k in ("billing", "technical", "general")}
    handlers["default"] = handlers["general"]
    return run_router("My login is broken and throws an error", classify, handlers, tracker=tracker)


def _self_test() -> None:
    r = demo()
    assert r.ok and r.trace.steps[0].detail == "technical", r.trace.steps[0].detail
    assert "debug" in r.output

    # Unknown category falls back to default but is flagged not-ok.
    tracker = CostTracker()
    fb = run_router(
        "hi",
        classifier=lambda t: "unknown",
        handlers={"default": lambda t: "fallback"},
        tracker=tracker,
    )
    assert (not fb.ok) and fb.output == "fallback"
    print("[OK] routing._self_test passed")


if __name__ == "__main__":
    _self_test()
    print(demo().trace.render())
