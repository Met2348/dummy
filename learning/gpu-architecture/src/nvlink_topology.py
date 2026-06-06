"""NVLink topology — DGX H100 (8-GPU NVSwitch) vs GB200 NVL72."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Topology:
    name: str
    n_gpus: int
    per_gpu_link_tb_s: float       # bidirectional
    diameter_hops: int             # max hops between any two GPUs
    bisection_tb_s: float          # cross-section bandwidth


TOPOLOGIES = {
    "dgx_a100_8":   Topology("DGX-A100 8x NVSwitch",   8, 0.6, 1, 4.8),
    "dgx_h100_8":   Topology("DGX-H100 8x NVSwitch",   8, 0.9, 1, 7.2),
    "gb200_nvl72": Topology("GB200 NVL72",            72, 1.8, 1, 129.6),
    "pcie_8":       Topology("PCIe-only 8 GPUs",       8, 0.064, 1, 0.5),
}


def allreduce_time_ms(topo: Topology, bytes_total: int) -> float:
    """Ring all-reduce time. 2*(N-1)/N * size / per_gpu_link_BW."""
    n = topo.n_gpus
    ring_factor = 2 * (n - 1) / n
    bw_gb_s = topo.per_gpu_link_tb_s * 1000
    return (bytes_total / 1e9) / bw_gb_s * 1000 * ring_factor


def compare(bytes_total: int) -> list[dict]:
    rows = []
    for k, t in TOPOLOGIES.items():
        rows.append({
            "topo": t.name,
            "gpus": t.n_gpus,
            "allreduce_ms": round(allreduce_time_ms(t, bytes_total), 3),
            "bisection_tb_s": t.bisection_tb_s,
        })
    return rows


def _self_test() -> None:
    rows = compare(bytes_total=int(1e9))  # 1 GB all-reduce
    by_name = {r["topo"]: r for r in rows}
    assert by_name["DGX-H100 8x NVSwitch"]["allreduce_ms"] < by_name["DGX-A100 8x NVSwitch"]["allreduce_ms"]
    assert by_name["PCIe-only 8 GPUs"]["allreduce_ms"] > 10 * by_name["DGX-H100 8x NVSwitch"]["allreduce_ms"]
    print(f"[OK] nvlink_topology (H100 1GB allreduce {by_name['DGX-H100 8x NVSwitch']['allreduce_ms']}ms)")


if __name__ == "__main__":
    _self_test()
