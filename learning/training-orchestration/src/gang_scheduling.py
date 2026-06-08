"""Gang scheduling: all-or-nothing for distributed training."""
from __future__ import annotations
from common import Job, Node, JobState


def gang_assign(job: Job, nodes: list[Node]) -> bool:
    """Atomic: assign all N gpus, or none. No partial."""
    free = [n.n_gpus_free for n in nodes]
    needed = job.n_gpus
    plan = []
    for i, f in enumerate(free):
        if needed <= 0:
            break
        take = min(f, needed)
        if take > 0:
            plan.append((i, take))
            needed -= take
    if needed > 0:
        return False
    for nid, take in plan:
        nodes[nid].n_gpus_free -= take
        job.node_assignment.append(nid)
    return True


def gang_release_all(jobs: list[Job], nodes: list[Node]) -> None:
    """Topology-aware release: each rank's GPU returns to its assigned node."""
    for j in jobs:
        if not j.node_assignment:
            continue
        # Each rank takes 1 GPU; distribute n_gpus across assignment list
        per = j.n_gpus // max(1, len(j.node_assignment))
        for nid in j.node_assignment:
            nodes[nid].n_gpus_free = min(
                nodes[nid].n_gpus_total, nodes[nid].n_gpus_free + per
            )
        j.node_assignment = []


def starvation_check(queue: list[Job], capacity: int) -> list[Job]:
    """If any single job needs > capacity, it will starve forever."""
    return [j for j in queue if j.n_gpus > capacity]


def _self_test() -> None:
    from common import make_cluster
    cluster = make_cluster(8, 8)         # 64 GPUs

    j_big = Job(1, "alice", 64, 7200)
    j_med = Job(2, "bob",   16, 3600)

    # Gang assign big first
    assert gang_assign(j_big, cluster)
    assert sum(n.n_gpus_free for n in cluster) == 0
    # Bob can't fit
    assert not gang_assign(j_med, cluster)

    # Starvation: job needing > 64 starves
    j_huge = Job(3, "carol", 128, 1000)
    assert j_huge in starvation_check([j_huge], 64)
    print("[OK] gang_scheduling")


if __name__ == "__main__":
    _self_test()
