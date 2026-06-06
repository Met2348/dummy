"""Fault tolerance — MTBF + ckpt strategy = wasted time formula."""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass
class FaultModel:
    n_gpus: int
    per_gpu_mtbf_hours: float = 8760.0    # ~1 year per GPU
    network_mtbf_hours: float = 720.0     # 1 month per fabric

    def cluster_mtbf_hours(self) -> float:
        """Failure rate is additive across components."""
        gpu_rate = self.n_gpus / self.per_gpu_mtbf_hours
        net_rate = 1.0 / self.network_mtbf_hours
        total_rate = gpu_rate + net_rate
        return 1.0 / total_rate


def optimal_ckpt_interval(ckpt_cost_s: float, mtbf_hours: float) -> float:
    """T_opt = sqrt(2 * C * M) — classic Young's formula (1974)."""
    mtbf_s = mtbf_hours * 3600
    return math.sqrt(2 * ckpt_cost_s * mtbf_s)


def expected_wasted_pct(ckpt_cost_s: float, ckpt_interval_s: float,
                       mtbf_hours: float) -> float:
    mtbf_s = mtbf_hours * 3600
    # Per-ckpt overhead share of training time
    overhead_pct = ckpt_cost_s / ckpt_interval_s
    # Failures redo half interval on average
    failure_pct = (ckpt_interval_s / 2) / mtbf_s
    return 100 * (overhead_pct + failure_pct)


def _self_test() -> None:
    fm = FaultModel(n_gpus=1024)
    mtbf = fm.cluster_mtbf_hours()
    # 1024 GPU + 1 fabric → very low MTBF
    assert mtbf < 24, mtbf

    # ckpt cost 1s, MTBF 8.5h → optimal ~248s = ~4 min
    t_opt = optimal_ckpt_interval(1.0, 8.5)
    assert 200 < t_opt < 300, t_opt

    # Wasted with optimal
    w = expected_wasted_pct(1.0, t_opt, 8.5)
    assert w < 2.0, w
    # Wasted if interval is much longer than optimal
    w_bad = expected_wasted_pct(1.0, 3600, 8.5)
    assert w_bad > w, (w, w_bad)
    print(f"[OK] fault_tolerance (1024 GPU MTBF {mtbf:.1f}h, T_opt {t_opt:.0f}s, "
          f"wasted {w:.2f}%)")


if __name__ == "__main__":
    _self_test()
