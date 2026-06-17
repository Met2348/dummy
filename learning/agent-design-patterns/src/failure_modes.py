"""Agent anti-patterns — each as a broken demo + the fix.

These are the failures that don't throw exceptions; they quietly degrade
quality. Knowing their *shape* is half of agent design.

  1. runaway loop   — agent repeats the same action forever
  2. context rot    — irrelevant content dilutes retrieval/attention
  3. tool sprawl    — too many tools -> wrong tool picked
  4. silent failure — tool error swallowed, agent proceeds on garbage
"""
from __future__ import annotations

from common import est_tokens


# --- 1. Runaway loop -------------------------------------------------------
def runaway_without_guard(policy_action, max_real_steps=1000):
    """A loop with no progress check will run to its hard cap."""
    steps = 0
    history = []
    while steps < max_real_steps:
        history.append(policy_action)
        steps += 1
    return steps  # always hits the cap


def loop_with_progress_guard(actions, no_progress_limit=2):
    """FIX: stop once the last `no_progress_limit` actions are identical."""
    history = []
    for a in actions:
        history.append(a)
        if len(history) >= no_progress_limit and len(set(history[-no_progress_limit:])) == 1:
            return {"stopped": True, "reason": "no progress", "steps": len(history)}
    return {"stopped": False, "steps": len(history)}


# --- 2. Context rot --------------------------------------------------------
def retrieve_naive(query, docs):
    """No filtering: everything goes in, signal drowns in noise."""
    return docs


def retrieve_relevant(query, docs, k=2):
    """FIX: rank by keyword overlap, keep top-k. Less context, more signal."""
    q = set(query.lower().split())

    def score(d):
        return len(q & set(d.lower().split()))

    return sorted(docs, key=score, reverse=True)[:k]


# --- 3. Tool sprawl --------------------------------------------------------
def pick_tool(query, tools):
    """Naive: first tool whose name appears in the query. With 30 tools whose
    names overlap, this misfires constantly."""
    for name in tools:
        if name in query:
            return name
    return tools[0]  # arbitrary fallback — the sprawl trap


def pick_tool_curated(query, tools, aliases):
    """FIX: a small curated set + intent aliases instead of 30 raw names."""
    q = query.lower()
    for intent, name in aliases.items():
        if intent in q:
            return name
    return None  # explicit "no tool" beats a wrong tool


# --- 4. Silent failure -----------------------------------------------------
def run_tool_swallowing(tool, *a):
    """Anti-pattern: catch-all that hides the error and returns a fake value."""
    try:
        return tool(*a)
    except Exception:
        return ""  # agent now reasons on an empty string, none the wiser


def run_tool_surfacing(tool, *a):
    """FIX: return a structured error the agent (and trace) can SEE and react to."""
    try:
        return {"ok": True, "value": tool(*a)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _self_test() -> None:
    # 1. runaway vs guarded
    assert runaway_without_guard("noop", max_real_steps=50) == 50
    guarded = loop_with_progress_guard(["a", "b", "x", "x", "x", "x"])
    assert guarded["stopped"] and guarded["steps"] == 4, guarded

    # 2. context rot: relevant retrieval beats dumping everything
    docs = ["python error stack trace", "cooking recipe", "billing refund policy",
            "python import bug fix"]
    naive = retrieve_naive("python bug", docs)
    good = retrieve_relevant("python bug", docs, k=2)
    assert len(naive) == 4 and len(good) == 2
    assert all("python" in d for d in good), good

    # 3. tool sprawl: curated aliases pick the right tool; naive misfires
    tools = ["search", "search_web", "search_code", "send_email"]
    assert pick_tool("please search the email", tools) == "search"  # wrong! grabbed 'search'
    curated = pick_tool_curated(
        "please email the team", tools, aliases={"email": "send_email", "code": "search_code"}
    )
    assert curated == "send_email"

    # 4. silent vs surfaced failure
    def boom():
        raise ValueError("disk full")

    assert run_tool_swallowing(boom) == ""  # the dangerous case
    surfaced = run_tool_surfacing(boom)
    assert (not surfaced["ok"]) and "disk full" in surfaced["error"]
    print("[OK] failure_modes._self_test passed")


if __name__ == "__main__":
    _self_test()
