"""
open-science-and-communication 专题环境自检.
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
    import open_science_audit as osa

    good = osa.blank_release_plan()
    good["interdisciplinary_glossary"] = "占位: 跨学科术语对照表说明, 足够长用于冒烟测试。"
    good["public_communication"] = "占位: 公众沟通材料说明, 足够长用于冒烟测试。"
    good["preregistration"] = "占位: 预注册计划说明, 足够长用于冒烟测试。"
    good["artifact_release_plan"] = "占位: 代码/数据发布规范说明, 足够长用于冒烟测试。"
    good["social_media_boundary"] = "占位: 学术社交媒体边界说明, 足够长用于冒烟测试。"
    bad = osa.blank_release_plan()
    osa_ok = (
        osa.audit(good)["ready"]
        and not osa.audit(bad)["ready"]
        and len(osa.audit(bad)["issues"]) == len(osa.SECTIONS)
    )
    print(f"  [{'OK ' if osa_ok else 'FAIL'}] open_science_audit (完整计划通过 / 空计划被拦截)")

    return osa_ok


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
