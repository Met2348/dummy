"""
research-presentation 专题环境自检.
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

REQUIRED = ["nbformat"]


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
    import talk_planner as tp
    import pitch_kit as pk

    p = tp.plan(12, "Robust-DPO 优势随噪声扩大")
    plan_ok = len(p["parts"]) == 4 and abs(sum(x["minutes"] for x in p["parts"]) - 12) < 0.5
    print(f"  [{'OK ' if plan_ok else 'FAIL'}] talk_planner.plan (4 部分, 时间合计≈12)")

    bad = tp.plan(10, "")
    a = tp.audit(bad)
    audit_ok = not a["ok"] and any("takeaway" in i for i in a["issues"])
    print(f"  [{'OK ' if audit_ok else 'FAIL'}] talk_planner.audit (抓出空 takeaway)")

    r = pk.check_pitch("我做带噪声标签的 DPO 鲁棒性。", "10s", audience="layperson")
    pk_ok = len(r["jargon"]) > 0  # 应检出黑话
    print(f"  [{'OK ' if pk_ok else 'FAIL'}] pitch_kit 黑话检测 (检出 {r['jargon']})")

    return plan_ok and audit_ok and pk_ok


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
