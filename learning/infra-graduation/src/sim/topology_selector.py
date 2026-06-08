"""Topology selector: given budget + model, recommend a blueprint."""
from __future__ import annotations
from sim.common import (
    ClusterBlueprint, GPU_CATALOG, FABRIC_CATALOG, STORAGE_CATALOG)
from sim.time_to_train import ModelSpec, time_to_train_days
from sim.cost_model import total_cost_3y


def candidates() -> list[ClusterBlueprint]:
    out = []
    for gpu in ["H100", "H200", "B200"]:
        for n_nodes in [16, 64, 256, 1024]:
            for fab in ["ib_ndr", "ib_xdr"]:
                for st in ["lustre", "gpfs"]:
                    out.append(ClusterBlueprint(
                        n_nodes=n_nodes, gpus_per_node=8,
                        gpu=GPU_CATALOG[gpu],
                        fabric=FABRIC_CATALOG[fab],
                        storage=STORAGE_CATALOG[st]))
    return out


def select(model: ModelSpec, budget_usd: float, max_days: float) -> ClusterBlueprint | None:
    """Pick cheapest blueprint meeting time and budget."""
    feasible = []
    for bp in candidates():
        cost = total_cost_3y(bp)
        if cost["tco_3y_m"] * 1e6 > budget_usd:
            continue
        est = time_to_train_days(model, bp)
        if est["wall_days"] > max_days:
            continue
        feasible.append((cost["tco_3y_m"], bp, est["wall_days"]))
    if not feasible:
        return None
    feasible.sort(key=lambda x: (x[0], x[2], id(x[1])))
    return feasible[0][1]


def _self_test() -> None:
    # Smaller demo model so a moderately-sized cluster fits the deadline
    demo = ModelSpec("demo-7B", 7, 500)   # 500B tokens
    pick = select(demo, budget_usd=20e6, max_days=60)
    assert pick is not None
    assert pick.gpu.name in ("H100", "H200", "B200")
    # Tight budget plus tight time gives no solution.
    impossible = select(demo, budget_usd=100_000, max_days=1)
    assert impossible is None
    print(f"[OK] topology_selector (picked {pick.total_gpus()}x {pick.gpu.name} "
          f"+ {pick.fabric.name})")


if __name__ == "__main__":
    _self_test()
