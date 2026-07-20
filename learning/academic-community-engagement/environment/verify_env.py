"""
academic-community-engagement 专题环境自检.
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
    import engagement_scorer as es

    a = es.blank_engagement("测试邀约A")
    for key in es.DIMENSIONS:
        a["scores"][key] = {"score": 3, "note": "占位依据"}
    es_ok = es.audit(a)["ready"] and es.total(a) == 12
    print(f"  [{'OK ' if es_ok else 'FAIL'}] engagement_scorer (打分完整性自检 + 总分计算)")

    bad = es.blank_engagement("测试邀约B(敷衍)")
    bad["scores"]["reciprocity"] = {"score": 5, "note": ""}
    bad_ok = not es.audit(bad)["ready"]
    print(f"  [{'OK ' if bad_ok else 'FAIL'}] engagement_scorer (敷衍打分被拦截)")

    return es_ok and bad_ok


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
