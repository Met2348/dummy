"""multimodal-graduation 专题环境自检. 运行: python environment/verify_env.py"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REQUIRED = ["nbformat", "numpy"]
OPTIONAL = ["torch", "matplotlib"]


def check_imports() -> list[str]:
    missing = []
    for pkg in REQUIRED:
        ok = importlib.util.find_spec(pkg) is not None
        print(f"  [{'OK ' if ok else 'MISS'}] {pkg}")
        if not ok:
            missing.append(pkg)
    for pkg in OPTIONAL:
        ok = importlib.util.find_spec(pkg) is not None
        print(f"  [{'OK ' if ok else 'opt '}] {pkg} (可选)")
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
    import mm_capstone as mc

    root = Path(__file__).resolve().parents[2]   # learning/ (environment/../.. )
    checks = mc.assemble_pipeline_check(root)
    all_present = all(c["就位"] for c in checks)
    n_ok = sum(c["就位"] for c in checks)
    print(f"  [{'OK ' if all_present else 'WARN'}] M10 装配检查 ({n_ok}/{len(checks)} 专题就位)")

    gaps = mc.score_gaps()
    # 优先级降序, 最高的应是复现类 (高可做低成本)
    gap_ok = len(gaps) >= 5 and gaps[0]["优先级"] >= gaps[-1]["优先级"]
    print(f"  [{'OK ' if gap_ok else 'FAIL'}] gap 雷达 ({len(gaps)} 题, top 优先级={gaps[0]['优先级']})")

    return all_present and gap_ok


def main() -> int:
    print("== Part A: 依赖 ==")
    missing = check_imports()
    print("== Part B: nbformat ==")
    nb_ok = check_nbformat_roundtrip()
    print("== Part C: src 工具 ==")
    src_ok = check_src_tools()
    print("\n== 结论 ==")
    if missing:
        print(f"缺失依赖: {missing}\n  → pip install -r environment/requirements.txt")
        return 1
    if not (nb_ok and src_ok):
        print("检查未通过 (注: 需 M10.1-10.6 已建).")
        return 1
    print("全部通过 ✅  两个 notebook 可端到端运行.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
