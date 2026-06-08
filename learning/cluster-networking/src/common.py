"""Network primitives: link, switch, fabric."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Link:
    name: str
    bw_gb_s: float       # one direction
    latency_us: float


LINKS = {
    "pcie5_x16":  Link("PCIe Gen5 x16",       64.0,   1.0),
    "nvlink4":    Link("NVLink 4 (per GPU)", 450.0,   0.5),    # 900 GB/s bidir
    "nvlink5":    Link("NVLink 5 (per GPU)", 900.0,   0.4),    # 1800 GB/s bidir
    "ib_ndr":     Link("IB NDR 400G",         50.0,   1.5),
    "ib_xdr":     Link("IB XDR 800G",        100.0,   1.2),
    "roce_400g":  Link("RoCEv2 400G",         50.0,   2.5),
    "eth_100g":   Link("Ethernet 100G",       12.5,   5.0),
}


def time_to_send(bytes_total: int, link: Link) -> float:
    """Time in microseconds, including 1-way latency."""
    transfer_us = (bytes_total / 1e9) / link.bw_gb_s * 1e6
    return link.latency_us + transfer_us


def _self_test() -> None:
    # 1 GB on NVLink 4 is about 2.22 ms.
    t = time_to_send(int(1e9), LINKS["nvlink4"])
    assert 2000 < t < 2500, t
    # IB NDR for the same payload is about 20 ms.
    t_ib = time_to_send(int(1e9), LINKS["ib_ndr"])
    assert t_ib > 8 * t, (t, t_ib)
    print(f"[OK] cluster_networking.common (1GB NVLink {t:.0f}us, IB {t_ib:.0f}us)")


if __name__ == "__main__":
    _self_test()
