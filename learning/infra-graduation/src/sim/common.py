"""Cluster blueprint dataclasses."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class GPUSpec:
    name: str
    bf16_tflops: float
    fp8_tflops: float
    hbm_gb: int
    hbm_tb_s: float
    nvlink_tb_s: float
    tdp_w: int
    price_usd: int          # H100 ~$30k, B200 ~$40k retail


GPU_CATALOG = {
    "A100": GPUSpec("A100-80G", 312, 0,    80, 2.0,  0.6, 400, 15000),
    "H100": GPUSpec("H100",     989, 1979, 80, 3.35, 0.9, 700, 30000),
    "H200": GPUSpec("H200",     989, 1979, 141, 4.8,  0.9, 700, 35000),
    "B200": GPUSpec("B200",    2250, 4500, 192, 8.0,  1.8, 1000, 40000),
}


@dataclass(frozen=True)
class FabricSpec:
    name: str
    per_node_bw_gb_s: float
    latency_us: float
    has_sharp: bool


FABRIC_CATALOG = {
    "ib_ndr":  FabricSpec("IB NDR 400G",  50.0,  1.5, True),
    "ib_xdr":  FabricSpec("IB XDR 800G",  100.0, 1.2, True),
    "roce":    FabricSpec("RoCEv2 400G",  50.0,  2.5, False),
    "eth":     FabricSpec("Eth 100G",     12.5,  5.0, False),
}


@dataclass(frozen=True)
class StorageSpec:
    name: str
    bw_gb_s: float
    cap_pb: float


STORAGE_CATALOG = {
    "lustre": StorageSpec("Lustre OSS pool",  400, 20),
    "gpfs":   StorageSpec("GPFS",             600, 50),
    "nvme":   StorageSpec("Local NVMe RAID0", 100,  0.25),
}


@dataclass
class ClusterBlueprint:
    n_nodes: int
    gpus_per_node: int
    gpu: GPUSpec
    fabric: FabricSpec
    storage: StorageSpec

    def total_gpus(self) -> int:
        return self.n_nodes * self.gpus_per_node

    def total_hbm_gb(self) -> int:
        return self.total_gpus() * self.gpu.hbm_gb

    def total_bf16_pflops(self) -> float:
        return self.total_gpus() * self.gpu.bf16_tflops / 1000.0

    def capex_usd(self) -> int:
        return self.total_gpus() * self.gpu.price_usd


def _self_test() -> None:
    bp = ClusterBlueprint(
        n_nodes=64, gpus_per_node=8,
        gpu=GPU_CATALOG["H100"],
        fabric=FABRIC_CATALOG["ib_ndr"],
        storage=STORAGE_CATALOG["lustre"],
    )
    assert bp.total_gpus() == 512
    assert bp.total_hbm_gb() == 512 * 80
    assert 500 < bp.total_bf16_pflops() < 550   # ~506 PFLOPS
    assert bp.capex_usd() == 512 * 30000        # ~15M USD
    print(f"[OK] sim.common (512x H100 = {bp.total_bf16_pflops():.0f} PFLOPS, "
          f"${bp.capex_usd()/1e6:.1f}M)")


if __name__ == "__main__":
    _self_test()
