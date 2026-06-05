"""OpenAI Swarm-style: hand-off via function return."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class SwarmAgent:
    name: str
    instructions: str
    functions: list[Callable[[dict], Any]] = field(default_factory=list)


class Swarm:
    def __init__(self):
        self.history: list[dict] = []

    def run(self, agent: SwarmAgent, message: str, context: dict | None = None,
            max_turns: int = 5) -> dict:
        active = agent
        ctx = dict(context or {})
        self.history.append({"from": "user", "to": active.name, "content": message})

        for _ in range(max_turns):
            result = self._step(active, message, ctx)
            if isinstance(result, SwarmAgent):
                self.history.append({"from": active.name, "type": "transfer", "to": result.name})
                active = result
                continue
            if isinstance(result, dict) and "transfer_to" in result:
                target = result["transfer_to"]
                self.history.append({"from": active.name, "type": "transfer", "to": target.name})
                active = target
                continue
            self.history.append({"from": active.name, "content": str(result)})
            return {"agent": active.name, "result": result, "history": self.history}
        return {"agent": active.name, "result": "(max_turns)", "history": self.history}

    def _step(self, agent: SwarmAgent, message: str, ctx: dict) -> Any:
        for fn in agent.functions:
            if not getattr(fn, "swarm_keyword", None):
                continue
            if fn.swarm_keyword in message.lower():
                return fn(ctx)
        return f"{agent.name} replies: {agent.instructions} (msg={message})"


def hand_off_keyword(keyword: str):
    def decorator(fn):
        fn.swarm_keyword = keyword
        return fn
    return decorator


def _self_test() -> None:
    agent_b = SwarmAgent(name="B", instructions="I am B")
    agent_c = SwarmAgent(name="C", instructions="I am C")

    @hand_off_keyword("billing")
    def to_b(ctx): return agent_b

    @hand_off_keyword("tech")
    def to_c(ctx): return agent_c

    agent_a = SwarmAgent(name="A", instructions="triage", functions=[to_b, to_c])
    swarm = Swarm()

    r = swarm.run(agent_a, "I have a billing question")
    assert r["agent"] == "B"
    assert any(h.get("type") == "transfer" for h in r["history"])

    swarm2 = Swarm()
    r2 = swarm2.run(agent_a, "I have a tech question")
    assert r2["agent"] == "C"

    swarm3 = Swarm()
    r3 = swarm3.run(agent_a, "Just hello")
    assert r3["agent"] == "A"
    print("[OK] swarm_mock._self_test passed")


if __name__ == "__main__":
    _self_test()
