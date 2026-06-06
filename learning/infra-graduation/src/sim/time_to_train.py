"""Estimate end-to-end time to train an N-param model on a blueprint."""
from __future__ import annotations
from dataclasses import dataclass
from sim.common import ClusterBlueprint
import math


@dataclass
class ModelSpec:
    name: str
    n_params: int                 # billions
    n_tokens: int                 # billions for training
    dtype_bytes: int = 2          # BF16


# Per Chinchilla and Llama-3 disclosures: 6 * params * tokens FLOPs
def total_flops(model: ModelSpec) -> int:
    return 6 * model.n_params * 1_000_000_000 * model.n_tokens * 1_000_000_000


def step_compute_s(model: ModelSpec, bp: ClusterBlueprint,
                   util_pct: float = 0.40) -> float:
    """Seconds per training step assuming `util_pct` of peak BF16."""
    flops_per_step = 6 * model.n_params * 1e9 * 1_000_000      # 1M tokens/step batch
    peak = bp.total_bf16_pflops() * 1e15
    return flops_per_step / (peak * util_pct)


def step_comm_s(model: ModelSpec, bp: ClusterBlueprint) -> float:
    """All-reduce gradient time per step (simplified ring on fabric)."""
    grad_bytes = model.n_params * 1e9 * model.dtype_bytes
    # Ring all-reduce: 2*(N-1)/N * size / BW (use fabric per-node BW)
    n = bp.n_nodes
    bw = bp.fabric.per_node_bw_gb_s * 1e9
    factor = 2 * (n - 1) / n
    return grad_bytes / bw * factor


def time_to_train_days(model: ModelSpec, bp: ClusterBlueprint,
                       util_pct: float = 0.40,
                       overhead_factor: float = 1.25) -> dict:
    """Wall time estimate. overhead_factor covers comm/overlap/ckpt/data-stall.

    Real comm cost is much lower than naive all-reduce because TP/PP/FSDP
    shrink the DP gradient and modern fabrics overlap >80% of comm with compute.
    Empirical 2024-2025 well-engineered runs report 20-30% overhead total.
    """
    flops_total = total_flops(model)
    peak = bp.total_bf16_pflops() * 1e15
    pure_compute_s = flops_total / (peak * util_pct)
    n_steps = model.n_tokens * 1e9 / 1e6
    raw_comm_s = step_comm_s(model, bp) * n_steps        # informational
    total_s = pure_compute_s * overhead_factor
    return {
        "model": model.name,
        "cluster": f"{bp.total_gpus()}x {bp.gpu.name}",
        "pure_compute_days": round(pure_compute_s / 86400, 1),
        "raw_comm_days_unhidden": round(raw_comm_s / 86400, 2),
        "wall_days": round(total_s / 86400, 1),
        "util_pct": util_pct * 100,
    }


def _self_test() -> None:
    from sim.common import ClusterBlueprint, GPU_CATALOG, FABRIC_CATALOG, STORAGE_CATALOG
    bp = ClusterBlueprint(
        64, 8, GPU_CATALOG["H100"], FABRIC_CATALOG["ib_ndr"], STORAGE_CATALOG["lustre"])
    llama70b = ModelSpec("Llama-3-70B", 70, 15_000)        # 15T tokens
    est = time_to_train_days(llama70b, bp)
    # Sanity: 70B-15T on 512 H100 is ~1 year (real Llama-3 used 24k H100 for 1 week)
    assert 200 < est["wall_days"] < 800, est
    print(f"[OK] sim.time_to_train ({est['cluster']} → 70B-15T in {est['wall_days']}d)")


if __name__ == "__main__":
    _self_test()
