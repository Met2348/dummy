"""MLPerf Training mock: 5 tasks by time-to-target scoring."""
from __future__ import annotations
from dataclasses import dataclass
from sim.common import ClusterBlueprint, GPU_CATALOG, FABRIC_CATALOG, STORAGE_CATALOG
from sim.time_to_train import ModelSpec, time_to_train_days
from sim.cost_model import total_cost_3y


@dataclass
class MLPerfTask:
    name: str
    model: ModelSpec
    target_metric: str          # e.g. "BLEU 0.5" / "perplexity < 7.0"


TASKS = [
    MLPerfTask("llm-pretrain-8b",   ModelSpec("8B-pre", 8, 1000),   "loss<3.0"),
    MLPerfTask("llm-pretrain-70b",  ModelSpec("70B-pre", 70, 5000), "loss<2.5"),
    MLPerfTask("llm-finetune-70b",  ModelSpec("70B-ft", 70, 100),   "perplexity<5.0"),
    MLPerfTask("llm-pretrain-405b", ModelSpec("405B-pre", 405, 5000), "loss<2.1"),
    MLPerfTask("llm-finetune-405b", ModelSpec("405B-ft", 405, 50),  "perplexity<4.5"),
]


def score_task(task: MLPerfTask, bp: ClusterBlueprint) -> dict:
    t = time_to_train_days(task.model, bp)
    c = total_cost_3y(bp)
    return {
        "task": task.name,
        "days": t["wall_days"],
        "tco_per_run_m": round(t["wall_days"] / 365 * c["yearly_opex_m"]
                                + c["gpu_capex_m"] * t["wall_days"] / 1095, 3),
        # 3y depreciation
    }


def run_capstone_2() -> list[dict]:
    bp_h = ClusterBlueprint(64, 8, GPU_CATALOG["H100"],
                             FABRIC_CATALOG["ib_ndr"], STORAGE_CATALOG["lustre"])
    bp_b = ClusterBlueprint(64, 8, GPU_CATALOG["B200"],
                             FABRIC_CATALOG["ib_xdr"], STORAGE_CATALOG["lustre"])
    rows = []
    for task in TASKS:
        h = score_task(task, bp_h)
        b = score_task(task, bp_b)
        rows.append({
            "task": task.name,
            "h100_days": h["days"],
            "b200_days": b["days"],
            "speedup": round(h["days"] / max(b["days"], 0.01), 2),
        })
    return rows


def _self_test() -> None:
    rows = run_capstone_2()
    assert len(rows) == 5
    # B200 should consistently beat H100
    speedups = [r["speedup"] for r in rows]
    assert all(s >= 1.0 for s in speedups), speedups
    avg = sum(speedups) / len(speedups)
    print(f"[OK] mlperf_mock (5 tasks; avg B200/H100 speedup {avg:.2f}x)")


if __name__ == "__main__":
    _self_test()
