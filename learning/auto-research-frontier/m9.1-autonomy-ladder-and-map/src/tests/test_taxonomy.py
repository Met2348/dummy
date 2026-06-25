"""V2 测试：锁死 9.1 的诚实性——分级只认证据、hype gap 真被算出、
以及那条不舒服的洞见（自称 Scientist 的都自评、独立验证的最高只到 Analyst）真锁在数据上。
"""
from __future__ import annotations

import pathlib
import sys

# tests 在 src/tests/ → parent.parent = src/，把 src/ 加进 path 以 import 包
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from autonomy_map import (
    LEVEL_RANK, SYSTEM_CATALOG, STAGES, System, classify, classify_all, evidenced_level,
)


def _sys(**kw):
    base = dict(name="X", year="2025", claimed_level="tool", automates=(),
                human_sets_problem=True, independent_verification=False)
    base.update(kw)
    return System(**base)


def test_tool_when_single_stage():
    assert evidenced_level(_sys(automates=("writeup",))) == "tool"


def test_analyst_when_core_exec_but_human_sets_problem():
    s = _sys(automates=("experiment", "analysis"), human_sets_problem=True)
    assert evidenced_level(s) == "analyst"


def test_scientist_needs_self_problem_and_closed_loop():
    s = _sys(automates=("ideation", "experiment", "analysis"), human_sets_problem=False)
    assert evidenced_level(s) == "scientist"


def test_demote_when_human_sets_problem():
    """同样闭环，但问题由人给 → 从 scientist 降到 analyst（问题谁定是分水岭）。"""
    auto = ("ideation", "experiment", "analysis")
    assert evidenced_level(_sys(automates=auto, human_sets_problem=False)) == "scientist"
    assert evidenced_level(_sys(automates=auto, human_sets_problem=True)) == "analyst"


def test_hype_gap_detected():
    """自称 scientist 但证据只够 tool → gap=+2。"""
    c = classify(_sys(claimed_level="scientist", automates=("writeup",)))
    assert c.hype_gap == 2
    assert any("hype gap" in r for r in c.reasons)


def test_catalog_well_formed():
    for s in SYSTEM_CATALOG:
        assert s.claimed_level in ("tool", "analyst", "scientist")
        assert set(s.automates) <= set(STAGES), f"{s.name} 有非法阶段"
        assert len(set(s.automates)) == len(s.automates), f"{s.name} 阶段重复"


def test_no_independently_verified_system_reaches_scientist():
    """核心洞见①：目录里凡是有独立验证的，证据级别都还没到 Scientist。"""
    for c in classify_all():
        s = next(x for x in SYSTEM_CATALOG if x.name == c.name)
        if s.independent_verification:
            assert c.evidenced_level != "scientist", \
                f"{c.name} 既独立验证又算 Scientist？该更新教学结论了"


def test_every_scientist_is_self_verified():
    """核心洞见②：凡证据级别到 Scientist 的，结果都只靠自评。"""
    sci = [c for c in classify_all() if c.evidenced_level == "scientist"]
    assert sci, "目录里应至少有一个 Scientist 级系统"
    assert all(c.self_verified_only for c in sci)


if __name__ == "__main__":   # 直跑兜底（与 harness 的脚本回退一致）
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
