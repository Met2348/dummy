"""Capstone — Pattern Zoo.

ONE task ("free-text request -> structured ticket"), solved six ways with the
same building blocks, so you can see in one table how the *design choice*
changes cost, step count, and where each pattern shines. This is the whole
point of the topic: the model is fixed; the architecture is the lever.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ on path

from common import MockLLM, CostTracker
from patterns.prompt_chaining import run_chain
from patterns.routing import run_router
from patterns.parallelization import run_sectioning
from patterns.orchestrator_workers import run_orchestrator
from patterns.evaluator_optimizer import run_eval_opt
from patterns.autonomous_agent import run_agent


SAMPLE = (
    "Users can't log in on mobile after the latest update — the login button "
    "does nothing. It's blocking many customers, needs a fix ASAP."
)


# --- deterministic mock "extractors" (shared by every approach) -------------
def extract_title(text: str) -> str:
    t = text.lower()
    subject = "login" if ("log in" in t or "login" in t) else ("payment" if "pay" in t else "feature")
    platform = "mobile" if "mobile" in t else ("web" if "web" in t else "")
    verb = "Fix" if any(w in t for w in ("can't", "cannot", "broken", "does nothing",
                                         "bug", "error", "fail")) else "Add"
    return " ".join(p for p in (verb, platform, subject) if p)


def infer_priority(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ("asap", "blocking", "urgent", "critical", "outage")):
        return "high"
    if any(w in t for w in ("nice to have", "someday", "minor", "cosmetic")):
        return "low"
    return "medium"


def derive_labels(text: str) -> list:
    t, labels = text.lower(), []
    if "log" in t or "auth" in t:
        labels.append("auth")
    if "mobile" in t:
        labels.append("mobile")
    if any(w in t for w in ("bug", "error", "broken", "fail", "does nothing", "can't", "cannot")):
        labels.append("bug")
    if "after" in t and "update" in t:
        labels.append("regression")
    return labels or ["triage"]


def write_acceptance(text: str) -> str:
    return "Issue no longer reproduces in the reported scenario; a regression test is added."


_FNS = {
    "title": extract_title,
    "priority": infer_priority,
    "labels": derive_labels,
    "acceptance": write_acceptance,
}
_REQUIRED = ["title", "priority", "labels", "acceptance"]


@dataclass
class Ticket:
    title: str = ""
    priority: str = ""
    labels: list = field(default_factory=list)
    acceptance: str = ""

    def complete(self) -> bool:
        return bool(self.title and self.priority and self.labels and self.acceptance)

    def __str__(self) -> str:
        return f"[{self.priority}] {self.title} {self.labels}"


def to_ticket(d: dict) -> Ticket:
    return Ticket(d.get("title", ""), d.get("priority", ""),
                  d.get("labels") or [], d.get("acceptance", ""))


def _llm(tracker, reply=""):
    return MockLLM(responder=lambda p, **k: reply, tracker=tracker)


# --- six designs ------------------------------------------------------------
def via_chaining(text=SAMPLE):
    tracker = CostTracker()

    def make_step(fname):
        def step(d):
            _llm(tracker).complete(f"extract {fname}")
            d = dict(d)
            d[fname] = _FNS[fname](d["_text"])
            return d
        return step

    steps = [(f, make_step(f)) for f in _REQUIRED]
    r = run_chain({"_text": text}, steps, tracker=tracker)
    r.output = to_ticket(r.output)
    r.ok = r.output.complete()
    return r


def via_routing(text=SAMPLE):
    tracker = CostTracker()

    def classify(t, **k):
        _llm(tracker).complete("classify ticket type")
        tl = t.lower()
        if any(w in tl for w in ("bug", "error", "broken", "can't", "cannot", "does nothing")):
            return "bug"
        if any(w in tl for w in ("add", "feature", "support", "want")):
            return "feature"
        return "chore"

    def make_builder(kind):
        def b(t):
            _llm(tracker).complete(f"build {kind} ticket")
            return to_ticket({
                "title": extract_title(t), "priority": infer_priority(t),
                "labels": derive_labels(t) + [kind], "acceptance": write_acceptance(t),
            })
        return b

    handlers = {k: make_builder(k) for k in ("bug", "feature", "chore")}
    handlers["default"] = handlers["chore"]
    r = run_router(text, classify, handlers, tracker=tracker)
    r.ok = r.output.complete()
    return r


def via_sectioning(text=SAMPLE):
    tracker = CostTracker()

    def worker(field_name):
        _llm(tracker).complete(f"extract {field_name}")
        return (field_name, _FNS[field_name](text))

    r = run_sectioning(
        _REQUIRED, worker, aggregate=lambda rs: to_ticket(dict(rs)), tracker=tracker
    )
    r.ok = r.output.complete()
    return r


def via_orchestrator(text=SAMPLE):
    tracker = CostTracker()

    def planner(t, **k):
        # Dynamic: pick the fields this request supports. Here all four apply.
        _llm(tracker).complete("plan fields")
        return list(_REQUIRED)

    def worker(field_name):
        _llm(tracker).complete(f"fill {field_name}")
        return _FNS[field_name](text)

    def synth(t, results):
        return to_ticket(dict(results))

    r = run_orchestrator(text, planner, worker, synth, tracker=tracker)
    r.ok = r.output.complete()
    return r


def via_eval_opt(text=SAMPLE):
    tracker = CostTracker()
    state = {"d": {}}

    def gen(task, feedback):
        d = state["d"]
        for f in _REQUIRED:  # add one missing field per round
            if f not in d:
                d[f] = _FNS[f](text)
                break
        _llm(tracker).complete("generate ticket draft")
        return dict(d)

    def ev(d):
        missing = [f for f in _REQUIRED if not d.get(f)]
        return (not missing, "missing:" + ",".join(missing))

    r = run_eval_opt(text, gen, ev, max_iters=6, tracker=tracker)
    r.output = to_ticket(r.output)
    r.ok = r.output.complete()
    return r


def via_agent(text=SAMPLE):
    tracker = CostTracker()
    tools = {
        "extract_title": lambda: extract_title(text),
        "infer_priority": lambda: infer_priority(text),
        "derive_labels": lambda: derive_labels(text),
        "write_acceptance": lambda: write_acceptance(text),
    }
    order = ["extract_title", "infer_priority", "derive_labels", "write_acceptance"]

    def policy(state):
        _llm(tracker).complete("decide next action")  # each loop step = 1 model call
        done = dict(state["observations"])
        for name in order:
            if name not in done:
                return {"tool": name}
        return {"finish": to_ticket({
            "title": done["extract_title"], "priority": done["infer_priority"],
            "labels": done["derive_labels"], "acceptance": done["write_acceptance"],
        })}

    r = run_agent(text, policy, tools, max_steps=8, tracker=tracker)
    r.ok = bool(r.output) and r.output.complete()
    return r


APPROACHES = [
    ("prompt_chaining", via_chaining, "ordered, predictable subtasks"),
    ("routing", via_routing, "distinct input categories"),
    ("parallelization", via_sectioning, "independent facets, want speed"),
    ("orchestrator_workers", via_orchestrator, "subtask set unknown up front"),
    ("evaluator_optimizer", via_eval_opt, "clear criteria + iteration helps"),
    ("autonomous_agent", via_agent, "open-ended, needs flexibility (costliest)"),
]


def run_capstone():
    return [(name, fn(), when) for name, fn, when in APPROACHES]


def to_md(results) -> str:
    lines = [
        "# Capstone — Pattern Zoo (free-text request -> structured ticket)\n",
        f"> Task input: \"{SAMPLE}\"\n",
        "| Design | LLM calls | tokens | $ | steps | ticket ok | best when |",
        "|--------|----------:|-------:|--:|------:|:---------:|-----------|",
    ]
    for name, r, when in results:
        row = r.row()
        ok = "[PASS]" if r.ok else "[FAIL]"
        lines.append(
            f"| {name} | {row['calls']} | {row['tokens']} | {row['usd']:.5f} | "
            f"{row['steps']} | {ok} | {when} |"
        )
    lines.append("\n## Takeaway")
    lines.append("Same model, same extractors — the **design** alone moves cost "
                 "from ~2 calls (routing) to ~5 (agent). Start at the top of the "
                 "list; only move down when the task demands it.")
    return "\n".join(lines)


def _self_test() -> None:
    results = run_capstone()
    assert len(results) == 6
    for name, r, _ in results:
        assert r.ok, f"{name} produced an incomplete ticket"
        assert isinstance(r.output, Ticket) and r.output.complete()
    # The whole lesson: designs differ in cost.
    calls = {name: r.tracker.calls for name, r, _ in results}
    assert calls["routing"] < calls["autonomous_agent"], calls
    assert calls["prompt_chaining"] == 4 and calls["routing"] == 2, calls
    assert calls["autonomous_agent"] == 5, calls  # 4 tool steps + 1 finish decision
    print("[OK] capstone.pattern_zoo._self_test passed "
          f"(calls: {calls})")


if __name__ == "__main__":
    _self_test()
    print()
    print(to_md(run_capstone()))
