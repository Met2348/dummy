"""The other end of the spectrum — Autonomous Agent.

Not a fixed workflow: the model runs in an OPEN loop, choosing tools based on
accumulated observations, until it decides it's done (or hits a guard). Maximum
flexibility, maximum cost/risk. Anthropic's advice: reach for this only after
workflows prove insufficient — and always cap it (max_steps) and observe it.

    goal -> [policy picks action] -> [tool] -> observe -> (repeat) -> finish
"""
from __future__ import annotations

from common import Trace, PatternResult, CostTracker, preview


def run_agent(goal, policy, tools, max_steps=8, tracker=None):
    """policy: fn(state)->action; action is {"tool",("args")} or {"finish": result}.

    The max_steps cap is the single most important line here: an open loop with
    no ceiling is how agents burn budget / loop forever in production.
    """
    trace = Trace()
    state = {"goal": goal, "observations": [], "done": False, "result": None}
    for step in range(max_steps):
        action = policy(state)
        if "finish" in action:
            state["done"] = True
            state["result"] = action["finish"]
            trace.add("done", "finish", preview(action["finish"]))
            break
        name = action["tool"]
        obs = tools[name](**action.get("args", {}))
        state["observations"].append((name, obs))
        trace.add("tool", name, preview(obs))
    else:
        trace.add("loop", "max-steps", f"hit cap of {max_steps} without finishing")
    return PatternResult("autonomous_agent", state["result"], trace, tracker, ok=state["done"])


def demo():
    tracker = CostTracker()
    config = {"team_size": 4, "monthly_per_seat": 20}

    tools = {
        "read_config": lambda: config,
        "multiply": lambda a, b: a * b,
    }

    def policy(state):
        # Deterministic mock "reasoning": read config, compute, then finish.
        obs = dict(state["observations"])  # name -> last obs
        if "read_config" not in obs:
            return {"tool": "read_config"}
        cfg = obs["read_config"]
        if "multiply" not in obs:
            return {"tool": "multiply", "args": {"a": cfg["team_size"], "b": cfg["monthly_per_seat"]}}
        total = obs["multiply"]
        return {"finish": f"monthly budget = ${total}"}

    return run_agent("compute the monthly seat budget", policy, tools, max_steps=6, tracker=tracker)


def _self_test() -> None:
    r = demo()
    assert r.ok, "agent should finish"
    assert r.output == "monthly budget = $80", r.output
    assert r.trace.kinds() == ["tool", "tool", "done"]

    # Loop guard: a policy that never finishes must stop at max_steps, ok=False.
    tracker = CostTracker()
    runaway = run_agent(
        "spin",
        policy=lambda s: {"tool": "noop"},
        tools={"noop": lambda: "still going"},
        max_steps=3,
        tracker=tracker,
    )
    assert (not runaway.ok) and runaway.trace.kinds().count("tool") == 3
    assert runaway.trace.steps[-1].kind == "loop"
    print("[OK] autonomous_agent._self_test passed")


if __name__ == "__main__":
    _self_test()
    print(demo().trace.render())
