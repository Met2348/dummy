"""V2 测试：锁死 9.6 的诚实性——沙箱真拦越权、弱 rubric 被刷、强 rubric 只放行诚实、
信任自报的 rubric 是反的。
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mini_eval import (
    CANDIDATES, MALICIOUS_SRC, SafeExecError, heldout, run_eval, safe_exec,
    trust_print, visible_only,
)


def test_sandbox_runs_real_code():
    """诚实候选过沙箱后，classify 真能算对一个 held-out 点。"""
    ns, _ = safe_exec(CANDIDATES["honest"])
    fn = ns["classify"]
    assert fn(5, -2) == 1 and fn(-5, -5) == 0


def test_sandbox_blocks_forbidden():
    """越权候选（import os）被沙箱拦下。"""
    try:
        safe_exec(MALICIOUS_SRC)
        assert False, "本应抛 SafeExecError"
    except SafeExecError:
        pass


def test_honest_passes_real_rubrics():
    assert run_eval(CANDIDATES["honest"], visible_only)["passed"]
    assert run_eval(CANDIDATES["honest"], heldout)["passed"]


def test_hardcode_games_visible_but_fails_heldout():
    """硬编码：可见样本满分通过，held-out 露馅不过。"""
    assert run_eval(CANDIDATES["hardcode"], visible_only)["passed"]
    assert not run_eval(CANDIDATES["hardcode"], heldout)["passed"]


def test_print_fraud_only_passes_trust_print():
    """print 造假：骗过信任自报的 rubric，强 rubric 戳穿。"""
    assert run_eval(CANDIDATES["print-fraud"], trust_print)["passed"]
    assert not run_eval(CANDIDATES["print-fraud"], heldout)["passed"]
    assert not run_eval(CANDIDATES["print-fraud"], visible_only)["passed"]


def test_trust_print_rubric_is_inverted():
    """信任自报的 rubric 反着来：放行造假、却毙掉诚实（诚实没吹牛）。"""
    assert run_eval(CANDIDATES["print-fraud"], trust_print)["passed"]
    assert not run_eval(CANDIDATES["honest"], trust_print)["passed"]


def test_only_honest_passes_strong_rubric():
    """强 rubric(heldout) 在三个候选里只放行诚实。"""
    passed = {c: run_eval(src, heldout)["passed"] for c, src in CANDIDATES.items()}
    assert passed == {"honest": True, "hardcode": False, "print-fraud": False}


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
