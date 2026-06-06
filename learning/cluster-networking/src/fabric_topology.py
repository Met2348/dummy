"""Fat-tree / dragonfly fabric — model bisection bandwidth."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class FatTree:
    n_nodes: int            # endpoints
    radix: int              # switch port count
    oversubscription: float = 1.0   # 1.0 = full bisection, 2.0 = 2:1

    def bisection_gb_s(self, link_bw_gb_s: float) -> float:
        ideal = (self.n_nodes / 2) * link_bw_gb_s
        return ideal / self.oversubscription

    def n_switches_3tier(self) -> dict:
        leaf = self.n_nodes / (self.radix / 2)
        return {"leaf": int(leaf), "spine": int(leaf), "core": int(leaf / 2 + 1)}


@dataclass
class Dragonfly:
    n_groups: int
    nodes_per_group: int
    intra_bw_gb_s: float
    inter_bw_gb_s: float

    def n_nodes(self) -> int:
        return self.n_groups * self.nodes_per_group

    def avg_hops(self) -> float:
        """Avg hops for random traffic: 2 intra + 1 inter ≈ 3 expected."""
        return 3.0 if self.n_groups > 1 else 1.0


def _self_test() -> None:
    # 1024-GPU fat-tree, 64-port switch, full bisection on 50 GB/s links
    ft = FatTree(1024, 64, oversubscription=1.0)
    bis = ft.bisection_gb_s(50.0)
    assert bis == 512 * 50.0
    sw = ft.n_switches_3tier()
    assert sw["leaf"] == 32, sw

    # 2:1 oversubscription halves bisection
    ft2 = FatTree(1024, 64, oversubscription=2.0)
    assert ft2.bisection_gb_s(50.0) == 256 * 50.0

    # Dragonfly
    df = Dragonfly(n_groups=8, nodes_per_group=64, intra_bw_gb_s=100, inter_bw_gb_s=25)
    assert df.n_nodes() == 512
    assert df.avg_hops() == 3.0
    print(f"[OK] fabric_topology (1024-node FT bisection {bis:.0f} GB/s)")


if __name__ == "__main__":
    _self_test()
