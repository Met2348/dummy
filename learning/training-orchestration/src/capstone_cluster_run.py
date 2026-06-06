"""Capstone: 24h cluster simulation — 64 nodes × 8 GPU, mixed jobs + faults."""
from __future__ import annotations
from common import make_cluster, Job, JobState
from slurm_scheduler import fifo_with_backfill, release
from fault_tolerance import FaultModel, optimal_ckpt_interval


def synth_workload() -> list[Job]:
    """8 training jobs of varying size."""
    return [
        Job(1, "team-a",  64, 6*3600, priority=10, submitted_at=0),
        Job(2, "team-b",  32, 12*3600, priority=5, submitted_at=300),
        Job(3, "team-c",   8, 1*3600, priority=3, submitted_at=600),
        Job(4, "team-a", 128, 4*3600, priority=10, submitted_at=900),
        Job(5, "team-d",  16, 2*3600, priority=1, submitted_at=1200),
        Job(6, "team-e",  64, 8*3600, priority=8, submitted_at=1500),
        Job(7, "team-b",   8, 1*3600, priority=5, submitted_at=1800),
        Job(8, "team-c",  32, 6*3600, priority=3, submitted_at=2100),
    ]


def simulate_24h(n_nodes: int = 64, gpus_per_node: int = 8) -> dict:
    cluster = make_cluster(n_nodes, gpus_per_node)
    jobs = synth_workload()

    # Single scheduling pass at t=0 (mock — real Slurm runs continuously)
    scheduled = fifo_with_backfill(jobs, cluster)
    completed = [j for j in scheduled if j.state == JobState.RUNNING]
    n_unscheduled = len(jobs) - len(scheduled)

    # Fault model
    fm = FaultModel(n_gpus=n_nodes * gpus_per_node)
    mtbf = fm.cluster_mtbf_hours()
    t_opt = optimal_ckpt_interval(1.0, mtbf)

    used_gpus = sum(j.n_gpus for j in completed)
    total_gpus = n_nodes * gpus_per_node
    util = used_gpus / total_gpus

    return {
        "total_jobs": len(jobs),
        "scheduled": len(completed),
        "unscheduled": n_unscheduled,
        "gpu_utilization": round(util, 2),
        "cluster_mtbf_h": round(mtbf, 2),
        "optimal_ckpt_interval_s": round(t_opt, 0),
    }


def _self_test() -> None:
    r = simulate_24h()
    assert r["total_jobs"] == 8
    assert r["gpu_utilization"] > 0.5, r
    assert r["cluster_mtbf_h"] > 1.0, r
    print(f"[OK] capstone_cluster_run ({r['scheduled']}/8 jobs, util {r['gpu_utilization']}, "
          f"MTBF {r['cluster_mtbf_h']}h, T_ckpt {r['optimal_ckpt_interval_s']}s)")


if __name__ == "__main__":
    _self_test()
