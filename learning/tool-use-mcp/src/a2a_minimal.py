"""Minimal A2A protocol - Agent Card + skill + task lifecycle."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class AgentCard:
    name: str
    version: str
    description: str
    skills: list[dict] = field(default_factory=list)
    url: str = ""


@dataclass
class A2ATask:
    id: str
    skill_id: str
    status: str  # "pending" | "running" | "completed" | "failed" | "cancelled"
    output: object = None
    error: str | None = None


class A2AAgent:
    def __init__(self, card: AgentCard):
        self.card = card
        self.skills: dict[str, Callable[[dict], object]] = {}
        self.tasks: dict[str, A2ATask] = {}
        self._next_id = 1

    def add_skill(self, skill_id: str, name: str, description: str,
                  func: Callable[[dict], object]) -> None:
        self.skills[skill_id] = func
        self.card.skills.append({
            "id": skill_id, "name": name, "description": description,
        })

    def send_task(self, skill_id: str, message: dict) -> str:
        task_id = f"t{self._next_id}"
        self._next_id += 1
        if skill_id not in self.skills:
            self.tasks[task_id] = A2ATask(
                id=task_id, skill_id=skill_id, status="failed",
                error=f"unknown skill: {skill_id}",
            )
            return task_id
        self.tasks[task_id] = A2ATask(id=task_id, skill_id=skill_id, status="running")
        try:
            output = self.skills[skill_id](message)
            self.tasks[task_id].status = "completed"
            self.tasks[task_id].output = output
        except Exception as e:  # noqa: BLE001
            self.tasks[task_id].status = "failed"
            self.tasks[task_id].error = str(e)
        return task_id

    def get_task(self, task_id: str) -> A2ATask | None:
        return self.tasks.get(task_id)


def _self_test() -> None:
    card = AgentCard(name="agent-A", version="1.0", description="test agent")
    agent = A2AAgent(card)
    agent.add_skill("greet", "Greet", "say hi", lambda m: f"hi {m.get('name','world')}")
    agent.add_skill("divide", "Divide", "divide", lambda m: m["a"] / m["b"])

    assert len(card.skills) == 2

    t1 = agent.send_task("greet", {"name": "Claude"})
    assert agent.get_task(t1).status == "completed"
    assert agent.get_task(t1).output == "hi Claude"

    t2 = agent.send_task("divide", {"a": 10, "b": 0})
    assert agent.get_task(t2).status == "failed"
    assert "division" in agent.get_task(t2).error.lower()

    t3 = agent.send_task("nope", {})
    assert agent.get_task(t3).status == "failed"
    assert "unknown skill" in agent.get_task(t3).error
    print("[OK] a2a_minimal._self_test passed")


if __name__ == "__main__":
    _self_test()
