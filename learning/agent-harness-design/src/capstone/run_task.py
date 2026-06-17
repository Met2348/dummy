"""Capstone — Mini-Harness in action.

Task: "Read the team config, compute the monthly seat budget, save a report."
We run it twice to show the harness machinery end-to-end:

  1. mode=ask  (writes approved) -> agent completes, report "saved"
  2. mode=readonly (writes denied) -> permission layer blocks the write, the
     failure is SURFACED, the agent finishes gracefully without it.

Then we persist the computed budget to file-based memory and reload it,
demonstrating cross-session durability.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ on path

from harness.model import MockModel, ModelResponse, ToolCall
from harness.tools import ToolRegistry
from harness.permissions import PermissionManager
from harness.memory import Memory
from mini_harness import Harness


CONFIG = {"team": "infra", "seats": 6, "price_per_seat": 25}


def build_registry(report_store: dict) -> ToolRegistry:
    reg = ToolRegistry()
    reg.add("read_config", "read the team seat config", lambda: dict(CONFIG), read_only=True)
    reg.add("multiply", "multiply two numbers", lambda a, b: a * b, read_only=True)

    def write_report(path, text):
        report_store[path] = text
        return f"{path} ({len(text)} chars)"

    reg.add("write_report", "save a markdown report to a path", write_report)  # NOT read-only
    return reg


def budget_brain(messages):
    """Deterministic 'reasoning': read -> compute -> write -> report status."""
    res = {m["name"]: m["content"] for m in messages if m.get("role") == "tool"}
    if "read_config" not in res:
        return ModelResponse("Reading the team config.", [ToolCall("read_config")])
    cfg = res["read_config"]["value"]
    if "multiply" not in res:
        return ModelResponse(
            "Computing monthly budget.",
            [ToolCall("multiply", {"a": cfg["seats"], "b": cfg["price_per_seat"]})],
        )
    total = res["multiply"]["value"]
    if "write_report" not in res:
        text = f"# {cfg['team']} budget\nMonthly seat budget = ${total}"
        return ModelResponse("Saving the report.",
                             [ToolCall("write_report", {"path": "report.md", "text": text})])
    wr = res["write_report"]
    if wr.get("ok"):
        return ModelResponse(f"Done. Report saved: {wr['value']}. Monthly budget = ${total}.")
    return ModelResponse(f"Budget computed = ${total}, but the report could not be saved "
                         f"(permission denied). Surfacing this instead of pretending success.")


def run_once(mode: str):
    store: dict = {}
    reg = build_registry(store)
    perms = PermissionManager(mode=mode, allow=["read_config", "multiply"])
    h = Harness(MockModel(budget_brain), reg, role="You are a budgeting assistant.",
                env={"cwd": "/work", "config_keys": ",".join(CONFIG)}, budget=1000,
                permissions=perms, max_turns=8)
    result = h.run("Read the team config, compute the monthly seat budget, and save a report.")
    return h, result, store


def run_capstone():
    return {"ask": run_once("ask"), "readonly": run_once("readonly")}


def _self_test() -> None:
    # 1. ask mode: writes approved -> full success, report stored.
    h, result, store = run_once("ask")
    assert "Done. Report saved" in result and "$150" in result, result
    assert "report.md" in store
    assert h.tracker.model_calls == 4 and h.tracker.tool_calls == 3, h.tracker.summary()
    assert h.trace.count("perm") == 3

    # 2. readonly: the write is denied, surfaced, agent finishes gracefully.
    h2, result2, store2 = run_once("readonly")
    assert "permission denied" in result2 and "$150" in result2, result2
    assert store2 == {}, "no write should have happened under readonly"
    assert any(s.kind == "perm" and "deny" in s.detail for s in h2.trace.spans)

    # 3. memory durability across "sessions".
    d = tempfile.mkdtemp(prefix="harness_cap_")
    path = os.path.join(d, "mem.json")
    Memory(path).set("last_budget", 150)
    assert Memory(path).get("last_budget") == 150
    os.remove(path)
    os.rmdir(d)
    print("[OK] capstone.run_task._self_test passed "
          f"(ask: {h.tracker.summary()['usd']:.5f} usd, 3 tools; readonly: write blocked)")


def to_report() -> str:
    h, result, store = run_once("ask")
    h2, result2, _ = run_once("readonly")
    out = ["# Capstone — Mini-Harness run\n",
           "## Run 1: mode=ask (writes approved)\n",
           h.report(),
           f"\nFINAL: {result}\n",
           "\n## Run 2: mode=readonly (writes denied)\n",
           h2.report(),
           f"\nFINAL: {result2}\n",
           "\n## Takeaway",
           "Same model + same tools; the **permission mode** alone decides whether the "
           "side effect happens. The denied write is surfaced as a tool error the agent "
           "reacts to — not swallowed."]
    return "\n".join(out)


if __name__ == "__main__":
    _self_test()
    print()
    print(to_report())
