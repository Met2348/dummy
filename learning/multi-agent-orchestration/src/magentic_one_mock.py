"""Magentic-One style: Orchestrator + workers + 2 ledgers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Worker:
    name: str
    keywords: list[str]
    execute_fn: Callable[[str], str]


@dataclass
class TaskLedger:
    goal: str = ""
    facts: list[str] = field(default_factory=list)
    guesses: list[str] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    final: str = ""


@dataclass
class ProgressLedger:
    done: bool = False
    stuck: bool = False
    progressing: bool = True
    next_worker: str = ""
    next_instruction: str = ""


class Orchestrator:
    def __init__(self, workers: list[Worker]):
        self.workers = {w.name: w for w in workers}
        self.task_ledger = TaskLedger()
        self.progress_ledger = ProgressLedger()
        self.history: list[dict] = []
        self.stuck_counter = 0

    def _init_task_ledger(self, goal: str) -> None:
        self.task_ledger.goal = goal
        self.task_ledger.facts = [f"Goal: {goal}"]
        self.task_ledger.plan = list(self.workers.keys())

    def _update_progress(self) -> None:
        completed = {h["worker"] for h in self.history}
        remaining = [p for p in self.task_ledger.plan if p not in completed]
        if not remaining:
            self.progress_ledger.done = True
            self.progress_ledger.next_worker = ""
            return
        if self.stuck_counter >= 2:
            self.progress_ledger.stuck = True
        self.progress_ledger.done = False
        self.progress_ledger.next_worker = remaining[0]
        self.progress_ledger.next_instruction = f"Address: {self.task_ledger.goal}"

    def _replan(self) -> None:
        self.task_ledger.plan = list(reversed(list(self.workers.keys())))
        self.stuck_counter = 0

    def run(self, goal: str, max_rounds: int = 8) -> TaskLedger:
        self._init_task_ledger(goal)
        for _ in range(max_rounds):
            self._update_progress()
            if self.progress_ledger.done:
                self.task_ledger.final = " | ".join(
                    f"{h['worker']}:{h['result'][:30]}" for h in self.history
                )
                return self.task_ledger
            if self.progress_ledger.stuck:
                self._replan()
                continue
            w_name = self.progress_ledger.next_worker
            worker = self.workers[w_name]
            result = worker.execute_fn(self.progress_ledger.next_instruction)
            self.history.append({"worker": w_name, "result": result})
            self.task_ledger.facts.append(f"{w_name} reported: {result[:60]}")
        return self.task_ledger


def _self_test() -> None:
    workers = [
        Worker("WebSurfer", ["web", "search"], lambda i: "found ReAct paper"),
        Worker("FileSurfer", ["file"], lambda i: "wrote summary.md"),
        Worker("Coder", ["code"], lambda i: "def f(): pass"),
    ]
    orch = Orchestrator(workers)
    result = orch.run("Research ReAct and write summary", max_rounds=10)
    assert result.goal == "Research ReAct and write summary"
    assert len(orch.history) == 3, orch.history
    assert "WebSurfer" in result.final
    assert "Coder" in result.final
    print("[OK] magentic_one_mock._self_test passed")


if __name__ == "__main__":
    _self_test()
