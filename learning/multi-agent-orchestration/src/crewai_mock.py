"""CrewAI-style: Agent (role/goal/backstory) + Task + Crew (sequential / hierarchical)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class CrewAgent:
    role: str
    goal: str
    backstory: str
    execute_fn: Callable[[str, dict], str]


@dataclass
class Task:
    id: str
    description: str
    agent: CrewAgent
    expected_output: str = ""


@dataclass
class Crew:
    agents: list[CrewAgent]
    tasks: list[Task]
    process: str = "sequential"  # sequential | hierarchical

    def kickoff(self, inputs: dict) -> dict:
        if self.process == "sequential":
            return self._run_sequential(inputs)
        if self.process == "hierarchical":
            return self._run_hierarchical(inputs)
        raise ValueError(f"unknown process: {self.process}")

    def _run_sequential(self, inputs: dict) -> dict:
        context: dict = dict(inputs)
        for task in self.tasks:
            desc = task.description.format(**context) if "{" in task.description else task.description
            output = task.agent.execute_fn(desc, context)
            context[task.id] = output
        return {"final": context[self.tasks[-1].id], "context": context}

    def _run_hierarchical(self, inputs: dict) -> dict:
        context: dict = dict(inputs)
        manager = self.agents[0]
        for task in self.tasks:
            plan = manager.execute_fn(f"Plan for: {task.description}", context)
            context[f"{task.id}_plan"] = plan
            output = task.agent.execute_fn(task.description, context)
            context[task.id] = output
        return {"final": context[self.tasks[-1].id], "context": context}


def _self_test() -> None:
    researcher = CrewAgent(
        role="Researcher", goal="Find insights",
        backstory="Senior researcher",
        execute_fn=lambda d, c: f"insights: {c.get('topic','?')} is hot",
    )
    writer = CrewAgent(
        role="Writer", goal="Write blog",
        backstory="Editor",
        execute_fn=lambda d, c: f"blog about {c.get('topic','?')}: {c.get('t1','?')}",
    )

    t1 = Task(id="t1", description="Research {topic}", agent=researcher)
    t2 = Task(id="t2", description="Write a blog", agent=writer)

    crew = Crew(agents=[researcher, writer], tasks=[t1, t2], process="sequential")
    result = crew.kickoff({"topic": "LLM"})

    assert "insights" in result["context"]["t1"]
    assert "LLM" in result["final"]
    assert "insights" in result["final"]
    print("[OK] crewai_mock._self_test passed")


if __name__ == "__main__":
    _self_test()
