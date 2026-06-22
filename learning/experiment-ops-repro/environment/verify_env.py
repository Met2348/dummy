"""
experiment-ops-repro 专题环境自检.
运行: python environment/verify_env.py
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REQUIRED = ["nbformat", "numpy", "pandas"]


def check_imports() -> list[str]:
    missing = []
    for pkg in REQUIRED:
        ok = importlib.util.find_spec(pkg) is not None
        print(f"  [{'OK ' if ok else 'MISS'}] {pkg}")
        if not ok:
            missing.append(pkg)
    return missing


def check_nbformat_roundtrip() -> bool:
    import nbformat
    from nbformat.v4 import new_notebook, new_code_cell
    nb = new_notebook(cells=[new_code_cell("print('hi')")])
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "smoke.ipynb"
        nbformat.write(nb, p)
        back = nbformat.read(p, as_version=4)
    ok = back.cells[0].source == "print('hi')"
    print(f"  [{'OK ' if ok else 'FAIL'}] nbformat 读写往返")
    return ok


def check_src_tools() -> bool:
    src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(src))
    import exp_tracker as et
    import repro_check as rc

    with tempfile.TemporaryDirectory() as d:
        r = et.init("smoke", {"method": "DPO", "noise": 0.4, "seed": 1, "dataset": "x@v1"},
                    out_dir=d)
        r.log({"win_rate": 0.4})
        r.finish()
        runs = et.load_runs("smoke", out_dir=d)
    track_ok = len(runs) == 1 and "git_sha" in runs[0] and runs[0]["metrics"]["win_rate"] == 0.4
    print(f"  [{'OK ' if track_ok else 'FAIL'}] exp_tracker init/log/finish/load")

    rep = rc.audit(runs[0])
    audit_ok = rep["score"] >= 5   # seed/git/config/env/data/metrics 多数齐全
    print(f"  [{'OK ' if audit_ok else 'FAIL'}] repro_check.audit (score={rep['score']}/{rep['total']})")

    bad = rc.audit({"config": {"method": "DPO"}, "git_sha": "no-git", "metrics": {}})
    bad_ok = bad["score"] <= 1
    print(f"  [{'OK ' if bad_ok else 'FAIL'}] repro_check 识别糟糕记录 (score={bad['score']})")

    return track_ok and audit_ok and bad_ok


def main() -> int:
    print("== Part A: 依赖 import ==")
    missing = check_imports()
    print("== Part B: nbformat 往返 ==")
    nb_ok = check_nbformat_roundtrip()
    print("== Part C: src 工具冒烟 ==")
    src_ok = check_src_tools()

    print("\n== 结论 ==")
    if missing:
        print(f"缺失依赖: {missing}\n  → pip install -r environment/requirements.txt")
        return 1
    if not (nb_ok and src_ok):
        print("nbformat 或 src 工具检查未通过.")
        return 1
    print("全部通过 ✅  两个 notebook 可端到端运行.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
