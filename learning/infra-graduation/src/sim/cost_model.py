"""TCO model — capex + opex + power."""
from __future__ import annotations
from sim.common import ClusterBlueprint


def power_kw(bp: ClusterBlueprint, pue: float = 1.3) -> float:
    """GPU TDP + 30% non-GPU + PUE factor."""
    gpu_w = bp.total_gpus() * bp.gpu.tdp_w
    server_w = gpu_w * 1.3       # CPU + NIC + cooling fan
    return server_w * pue / 1000.0


def opex_yearly_usd(bp: ClusterBlueprint, electricity_usd_per_kwh: float = 0.10) -> float:
    kw = power_kw(bp)
    hours_per_year = 8760
    return kw * hours_per_year * electricity_usd_per_kwh


def total_cost_3y(bp: ClusterBlueprint) -> dict:
    capex = bp.capex_usd()
    opex_y = opex_yearly_usd(bp)
    # Fabric ~$2k/node, storage ~$50k/PB (enterprise Lustre)
    fabric_capex = bp.n_nodes * 2000
    storage_capex = int(bp.storage.cap_pb * 50_000)
    return {
        "gpu_capex_m": round(capex / 1e6, 1),
        "fabric_capex_k": round(fabric_capex / 1000, 0),
        "storage_capex_k": round(storage_capex / 1000, 0),
        "yearly_opex_m": round(opex_y / 1e6, 2),
        "tco_3y_m": round((capex + fabric_capex + storage_capex + 3 * opex_y) / 1e6, 1),
        "power_kw": round(power_kw(bp), 1),
    }


def _self_test() -> None:
    from sim.common import ClusterBlueprint, GPU_CATALOG, FABRIC_CATALOG, STORAGE_CATALOG
    bp = ClusterBlueprint(
        64, 8, GPU_CATALOG["H100"], FABRIC_CATALOG["ib_ndr"], STORAGE_CATALOG["lustre"])
    t = total_cost_3y(bp)
    # 512x H100 capex ~$15M, power ~600 kW
    assert t["gpu_capex_m"] > 10, t
    assert t["power_kw"] > 400, t
    print(f"[OK] sim.cost_model (512 H100 TCO 3y ${t['tco_3y_m']}M, "
          f"power {t['power_kw']}kW)")


if __name__ == "__main__":
    _self_test()
