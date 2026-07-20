"""
research-funding-and-resources 专题环境自检.
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
    import funding_plan_audit as fpa

    good = fpa.blank_funding_plan()
    good["budget_justification"] = "占位: 预算论证说明, 足够长用于冒烟测试。"
    good["compute_plan"] = "占位: 算力资源规划说明, 足够长用于冒烟测试。"
    good["data_management"] = "占位: 数据管理规划说明, 足够长用于冒烟测试。"
    good["collaboration_mou"] = "占位: 多机构合作协议说明, 足够长用于冒烟测试。"
    good["vendor_compliance"] = "占位: 第三方合规审查说明, 足够长用于冒烟测试。"
    bad = fpa.blank_funding_plan()
    fpa_ok = (
        fpa.audit(good)["ready"]
        and not fpa.audit(bad)["ready"]
        and len(fpa.reviewer_focus(bad)) == len(fpa.SECTIONS)
        and len(fpa.reviewer_focus(good)) == 0
    )
    print(f"  [{'OK ' if fpa_ok else 'FAIL'}] funding_plan_audit (完整计划通过 / 空计划被拦截 / reviewer_focus追问数量正确)")

    return fpa_ok


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
    print("全部通过 ✅  notebook 可端到端运行.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
