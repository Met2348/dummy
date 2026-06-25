"""V2 测试：锁死 9.8 capstone 的诚实性——天真评审被全部骗过、诚实过全守卫、
四种攻击各被对应守卫戳穿（毕业标准）。
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from integrity import (
    ATTACKS, audit, guard_dataset, guard_independent_review, guard_metric,
    guard_provenance, naive_accept, run_science,
)


def test_honest_report_passes_all_guards():
    a = audit(run_science(None))
    assert a["trustworthy"] is True
    assert a["failed"] == []


def test_naive_review_is_fooled_by_every_attack():
    """天真评审（信自评分）把诚实和四种造假一视同仁地收下——分不出真假。"""
    assert naive_accept(run_science(None))
    for atk in ATTACKS:
        assert naive_accept(run_science(atk)), f"{atk} 本应骗过天真评审"


def test_every_attack_is_caught_by_guards():
    """毕业标准：每种攻击都被守卫判为不可信。"""
    for atk in ATTACKS:
        assert audit(run_science(atk))["trustworthy"] is False, f"{atk} 漏网了"


def test_provenance_catches_hallucinated_ablation():
    assert not guard_provenance(run_science("halluc-ablation")).passed
    assert guard_provenance(run_science(None)).passed


def test_dataset_guard_catches_swap():
    assert not guard_dataset(run_science("dataset-swap")).passed
    assert guard_dataset(run_science(None)).passed


def test_metric_guard_catches_hardcode():
    """硬编码 acc=0.99 与独立复算不符 → 被抓。"""
    assert not guard_metric(run_science("hardcode-metric")).passed
    assert guard_metric(run_science(None)).passed


def test_independent_review_catches_gamed_self_review():
    assert not guard_independent_review(run_science("game-review")).passed
    assert guard_independent_review(run_science(None)).passed


def test_graduation_property():
    """诚实可信 AND 所有攻击不可信 —— 这就是 capstone 的毕业判定。"""
    honest_ok = audit(run_science(None))["trustworthy"]
    all_caught = all(not audit(run_science(a))["trustworthy"] for a in ATTACKS)
    assert honest_ok and all_caught


if __name__ == "__main__":   # 直跑兜底
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                fails += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"\n{'OK' if fails == 0 else f'{fails} FAILED'}")
    raise SystemExit(1 if fails else 0)
