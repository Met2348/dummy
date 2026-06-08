"""Capstone-1: end-to-end mini-cluster simulator.

Given 3 model sizes, 3 cluster sizes, and 2 fabrics = 18 scenarios,
report time-to-train and TCO.
"""
from __future__ import annotations
from sim.common import (
    ClusterBlueprint, GPU_CATALOG, FABRIC_CATALOG, STORAGE_CATALOG)
from sim.time_to_train import ModelSpec, time_to_train_days
from sim.cost_model import total_cost_3y


MODELS = [
    ModelSpec("8B-1T",   8,  1_000),       # demo: 1T tokens
    ModelSpec("70B-5T",  70, 5_000),       # demo: 5T tokens
    ModelSpec("405B-10T", 405, 10_000),    # demo: 10T tokens (real Llama-3 = 15T)
]

CONFIGS = [
    ("8x H100 + IB NDR",    1,   "H100", "ib_ndr"),
    ("64x H100 + IB NDR",   8,   "H100", "ib_ndr"),
    ("512x H100 + IB NDR",  64,  "H100", "ib_ndr"),
    ("512x B200 + IB XDR",  64,  "B200", "ib_xdr"),
    ("4096x H100 + IB XDR", 512, "H100", "ib_xdr"),
    ("4096x B200 + IB XDR", 512, "B200", "ib_xdr"),
]


def run() -> list[dict]:
    rows = []
    for model in MODELS:
        for label, n_nodes, gpu, fab in CONFIGS:
            bp = ClusterBlueprint(
                n_nodes=n_nodes, gpus_per_node=8,
                gpu=GPU_CATALOG[gpu],
                fabric=FABRIC_CATALOG[fab],
                storage=STORAGE_CATALOG["lustre"])
            t = time_to_train_days(model, bp)
            c = total_cost_3y(bp)
            rows.append({
                "model": model.name,
                "cluster": label,
                "days": t["wall_days"],
                "tco_3y_m": c["tco_3y_m"],
                "power_kw": c["power_kw"],
            })
    return rows


def _self_test() -> None:
    rows = run()
    assert len(rows) == 18
    # 8B-1T on 8 H100 takes years in this toy model.
    small = next(r for r in rows if r["model"] == "8B-1T" and "8x H100" in r["cluster"])
    assert small["days"] > 50, small
    # 405B-10T on 4096 H100 lands in a realistic educational range.
    big = next(r for r in rows if r["model"] == "405B-10T" and "4096x H100" in r["cluster"])
    assert 50 < big["days"] < 500, big
    # B200 same N should be faster than H100 (2-3x speedup)
    h_4k = next(r for r in rows if r["model"] == "405B-10T" and "4096x H100" in r["cluster"])
    b_4k = next(r for r in rows if r["model"] == "405B-10T" and "4096x B200" in r["cluster"])
    assert b_4k["days"] < h_4k["days"], (b_4k, h_4k)
    print(f"[OK] capstone_1 (18 scenarios; 8B/8GPU {small['days']}d; "
          f"405B/4k H100 {h_4k['days']}d -> B200 {b_4k['days']}d)")


if __name__ == "__main__":
    _self_test()
    print()
    print("Model         | Cluster                  | Days | TCO 3y ($M) | Power (kW)")
    print("--------------|--------------------------|-----:|------------:|----------:")
    for r in run():
        print(f"{r['model']:<13} | {r['cluster']:<24} | {r['days']:>4} | "
              f"{r['tco_3y_m']:>10} | {r['power_kw']:>8}")
