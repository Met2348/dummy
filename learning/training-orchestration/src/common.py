"""Cluster job model for Slurm/K8s/Ray style orchestration."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class JobState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PREEMPTED = "preempted"


@dataclass
class Job:
    job_id: int
    user: str
    n_gpus: int
    duration_s: float
    priority: int = 0
    state: JobState = JobState.PENDING
    submitted_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    node_assignment: list[int] = field(default_factory=list)

    def waited_s(self) -> float:
        if self.started_at is None:
            return 0.0
        return self.started_at - self.submitted_at


@dataclass
class Node:
    node_id: int
    n_gpus_total: int
    n_gpus_free: int


def make_cluster(n_nodes: int, gpus_per_node: int) -> list[Node]:
    return [Node(i, gpus_per_node, gpus_per_node) for i in range(n_nodes)]


def _self_test() -> None:
    cluster = make_cluster(16, 8)
    assert len(cluster) == 16
    assert sum(n.n_gpus_free for n in cluster) == 128
    j = Job(1, "alice", 8, 3600)
    assert j.state == JobState.PENDING
    assert j.waited_s() == 0.0
    print("[OK] training_orchestration.common")


if __name__ == "__main__":
    _self_test()
