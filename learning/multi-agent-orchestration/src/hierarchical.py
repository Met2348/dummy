"""Hierarchical pattern: 1 supervisor + N workers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class HWorker:
    name: str
    keywords: list[str]
    execute_fn: Callable[[str, dict], str]


@dataclass
class Supervisor:
    name: str
    workers: list[HWorker]
    history: list[dict] = field(default_factory=list)

    def route(self, query: str) -> str:
        q_low = query.lower()
        for w in self.workers:
            if any(kw in q_low for kw in w.keywords):
                return w.name
        return "FINISH"

    def run(self, query: str, max_steps: int = 10) -> dict:
        state = {"query": query, "results": {}}
        remaining = query
        for _ in range(max_steps):
            next_w = self.route(remaining)
            if next_w == "FINISH":
                break
            worker = next(w for w in self.workers if w.name == next_w)
            result = worker.execute_fn(remaining, state)
            self.history.append({"worker": next_w, "result": result})
            state["results"][next_w] = result
            for kw in worker.keywords:
                remaining = remaining.lower().replace(kw, "")
        state["final"] = self._synthesize(state)
        return state

    def _synthesize(self, state: dict) -> str:
        parts = [f"{name}: {res[:60]}" for name, res in state["results"].items()]
        return f"[supervisor {self.name}] " + " | ".join(parts)


def _self_test() -> None:
    research_w = HWorker(
        name="researcher",
        keywords=["research", "find"],
        execute_fn=lambda q, s: "found 3 papers",
    )
    code_w = HWorker(
        name="coder",
        keywords=["code", "implement", "write"],
        execute_fn=lambda q, s: "wrote 50 LOC",
    )
    sup = Supervisor(name="boss", workers=[research_w, code_w])

    result = sup.run("research ReAct and write code")
    assert "researcher" in result["results"]
    assert "coder" in result["results"]
    assert "found 3 papers" in result["final"]
    assert "wrote" in result["final"]
    print("[OK] hierarchical._self_test passed")


if __name__ == "__main__":
    _self_test()
