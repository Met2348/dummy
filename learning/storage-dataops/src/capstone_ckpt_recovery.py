"""Capstone: end-to-end ckpt + recovery cost for 7-day training run."""
from __future__ import annotations
from common import TIERS
from checkpoint import full_checkpoint, sharded_checkpoint, async_sharded


MODEL_BYTES = int(140e9)        # 70B BF16
N_GPUS = 512
TRAIN_HOURS = 7 * 24
CKPT_INTERVAL_HOURS = 1.0       # checkpoint every hour
MTBF_HOURS = 24.0               # mean time between failures (large cluster reality)


def total_overhead(strategy: str) -> dict:
    lustre = TIERS["lustre"]
    if strategy == "full":
        cost = full_checkpoint(MODEL_BYTES, N_GPUS, lustre)
    elif strategy == "sharded":
        cost = sharded_checkpoint(MODEL_BYTES, N_GPUS, lustre)
    elif strategy == "async":
        cost = async_sharded(MODEL_BYTES, N_GPUS, lustre)
    else:
        raise ValueError(strategy)

    n_ckpts = TRAIN_HOURS / CKPT_INTERVAL_HOURS
    blocking_overhead_s = (cost.sec if cost.blocking else 0.0) * n_ckpts

    # Recovery: on failure, reload last ckpt + redo half of CKPT_INTERVAL of work
    n_failures = TRAIN_HOURS / MTBF_HOURS
    recovery_s = n_failures * (cost.sec + (CKPT_INTERVAL_HOURS * 3600) / 2)

    total_s = blocking_overhead_s + recovery_s
    return {
        "strategy": strategy,
        "per_ckpt_s": round(cost.sec, 3),
        "n_ckpts": int(n_ckpts),
        "blocking_total_min": round(blocking_overhead_s / 60, 2),
        "recovery_total_h": round(recovery_s / 3600, 2),
        "wasted_pct": round(100 * total_s / (TRAIN_HOURS * 3600), 2),
    }


def _self_test() -> None:
    f = total_overhead("full")
    s = total_overhead("sharded")
    a = total_overhead("async")
    assert f["wasted_pct"] > s["wasted_pct"], (f, s)
    assert s["wasted_pct"] > a["wasted_pct"], (s, a)
    # All should be < 50% wasted for a sane setup
    assert a["wasted_pct"] < 20.0, a
    print(f"[OK] capstone_ckpt_recovery "
          f"(full {f['wasted_pct']}% vs async {a['wasted_pct']}% wasted)")


if __name__ == "__main__":
    _self_test()
    print()
    print("Strategy | per-ckpt | blocking (min) | recovery (h) | wasted %")
    print("---------|----------|----------------|--------------|---------")
    for s in ["full", "sharded", "async"]:
        r = total_overhead(s)
        print(f"{r['strategy']:<8} | {r['per_ckpt_s']:>8} | {r['blocking_total_min']:>14} | "
              f"{r['recovery_total_h']:>12} | {r['wasted_pct']:>7}%")
