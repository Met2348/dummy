"""Slurm-style FIFO with backfill."""
from __future__ import annotations
from common import Job, Node, JobState


def try_assign(job: Job, nodes: list[Node]) -> bool:
    """Find contiguous nodes (or pack across) to satisfy n_gpus."""
    remaining = job.n_gpus
    chosen = []
    for n in sorted(nodes, key=lambda x: -x.n_gpus_free):
        if remaining <= 0:
            break
        if n.n_gpus_free > 0:
            take = min(remaining, n.n_gpus_free)
            chosen.append((n, take))
            remaining -= take
    if remaining > 0:
        return False
    for n, take in chosen:
        n.n_gpus_free -= take
        job.node_assignment.append(n.node_id)
    return True


def release(job: Job, nodes: list[Node]) -> None:
    """Release GPUs (proportionally back to original assignment)."""
    n_assigned = len(job.node_assignment)
    if n_assigned == 0:
        return
    per_node = job.n_gpus // n_assigned
    rem = job.n_gpus % n_assigned
    for i, nid in enumerate(job.node_assignment):
        nodes[nid].n_gpus_free += per_node + (1 if i < rem else 0)


def fifo_with_backfill(queue: list[Job], nodes: list[Node], now: float = 0.0) -> list[Job]:
    """FIFO at head + backfill smaller jobs behind blocked head."""
    queue = sorted(queue, key=lambda j: (j.submitted_at, -j.priority))
    scheduled = []
    blocked = []
    for j in queue:
        if try_assign(j, nodes):
            j.started_at = now
            j.state = JobState.RUNNING
            scheduled.append(j)
        else:
            blocked.append(j)
    # Backfill: walk blocked, try opportunistic
    for j in blocked:
        if try_assign(j, nodes):
            j.started_at = now
            j.state = JobState.RUNNING
            scheduled.append(j)
    return scheduled


def _self_test() -> None:
    from common import make_cluster
    cluster = make_cluster(4, 8)        # 32 GPUs
    jobs = [
        Job(1, "alice", 24, 3600, submitted_at=0),
        Job(2, "bob",    8, 1800, submitted_at=1),
        Job(3, "carol",  4, 7200, submitted_at=2),
    ]
    sched = fifo_with_backfill(jobs, cluster)
    # alice 24 + bob 8 = 32 fills cluster; carol 4 doesn't fit
    assert len(sched) == 2, [j.job_id for j in sched]
    assigned_gpus = sum(s.n_gpus for s in sched)
    assert assigned_gpus == 32, assigned_gpus

    # After alice finishes, backfill carol
    alice = sched[0]
    release(alice, cluster)
    re_sched = fifo_with_backfill([jobs[2]], cluster)
    assert len(re_sched) == 1
    print(f"[OK] slurm_scheduler ({len(sched)}/3 initial + backfill carol)")


if __name__ == "__main__":
    _self_test()
