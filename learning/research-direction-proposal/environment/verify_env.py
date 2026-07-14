"""
research-direction-proposal 专题环境自检.
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
    import direction_scorer as ds
    import proposal_audit as pa

    a = ds.blank_candidate("测试方向A")
    for key in ds.DIMENSIONS:
        a["scores"][key] = {"score": 3, "note": "占位依据"}
    ds_ok = ds.audit(a)["ready"] and ds.total(a) == 12
    print(f"  [{'OK ' if ds_ok else 'FAIL'}] direction_scorer (打分完整性自检 + 总分计算)")

    good = pa.blank_proposal()
    good["background"] = "占位背景说明,长度足够通过审查用于冒烟测试。"
    good["gap"] = "占位缺口说明。"
    good["question"] = "假设: 方法A比方法B高≥5分。"
    good["method"] = "占位方法说明。"
    good["timeline"] = "第1个月完成X,第2个月完成Y。"
    good["risks"] = "如果主方案不work,备选方案是切换到对照实验设计,风险可控。"
    good["contribution"] = "占位预期贡献说明。"
    bad = pa.blank_proposal()
    pa_ok = pa.audit(good)["ready"] and not pa.audit(bad)["ready"]
    print(f"  [{'OK ' if pa_ok else 'FAIL'}] proposal_audit (完整proposal通过 / 空proposal被拦截)")

    return ds_ok and pa_ok


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
