"""Capstone: simulate gradient all-reduce across 4 fabrics × 4 cluster sizes."""
from __future__ import annotations
from common import LINKS
from allreduce_algos import ring_allreduce, halving_doubling
from sharp_inline import sharp_allreduce


SCENARIOS = [
    # (cluster size, link)
    (8,   LINKS["nvlink4"]),
    (8,   LINKS["pcie5_x16"]),
    (64,  LINKS["ib_ndr"]),
    (512, LINKS["ib_xdr"]),
]

# 70B model in BF16 → 140 GB gradient
GRADIENT_BYTES = int(140e9)


def run() -> list[dict]:
    rows = []
    for n_gpus, link in SCENARIOS:
        algos = {
            "ring": ring_allreduce(n_gpus, GRADIENT_BYTES, link),
            "halving_doubling": halving_doubling(n_gpus, GRADIENT_BYTES, link),
        }
        # SHARP only on IB/RoCE switches
        if "IB" in link.name or "RoCE" in link.name:
            algos["sharp"] = sharp_allreduce(n_gpus, GRADIENT_BYTES, link)
        best = min(algos, key=algos.get)
        rows.append({
            "n_gpus": n_gpus,
            "link": link.name,
            "best_algo": best,
            "best_time_s": round(algos[best] / 1e6, 2),
            "ring_time_s": round(algos["ring"] / 1e6, 2),
        })
    return rows


def _self_test() -> None:
    rows = run()
    assert len(rows) == 4
    # NVLink 8-GPU: halving_doubling beats ring per BW-step model (6 steps vs 14)
    nvl_row = next(r for r in rows if "NVLink" in r["link"])
    assert nvl_row["best_algo"] in ("ring", "halving_doubling"), nvl_row
    # 512-GPU IB should pick SHARP (in-network reduction = 2-step constant)
    big_row = next(r for r in rows if r["n_gpus"] == 512)
    assert big_row["best_algo"] == "sharp", big_row
    # PCIe ring is much slower than NVLink ring
    pcie_row = next(r for r in rows if "PCIe" in r["link"])
    assert pcie_row["ring_time_s"] > 5 * nvl_row["ring_time_s"], (pcie_row, nvl_row)
    print(f"[OK] capstone_cluster_sim "
          f"(NVLink 8-GPU {nvl_row['best_time_s']}s, IB 512 SHARP {big_row['best_time_s']}s)")


if __name__ == "__main__":
    _self_test()
    print()
    print("Cluster | Link            | Best algo       | Best (s) | Ring (s)")
    print("--------|-----------------|-----------------|----------|---------")
    for r in run():
        print(f"{r['n_gpus']:>7} | {r['link']:<15} | {r['best_algo']:<15} | "
              f"{r['best_time_s']:>8} | {r['ring_time_s']:>7}")
