"""generative-media-graduation 专题环境自检. 运行: python environment/verify_env.py"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REQUIRED = ["nbformat", "numpy", "matplotlib"]
OPTIONAL = ["torch"]


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
    import generative_capstone as gc

    # 装配检查: M13 全 6 专题 src 应全部 import + 烟测通过
    results = gc.assembly_check()
    n_ok = sum(1 for _, ok, _ in results)
    assembly_ok = n_ok == len(gc.STACK)
    print(f"  [{'OK ' if assembly_ok else 'FAIL'}] M13 全链装配 ({n_ok}/{len(gc.STACK)} 专题烟测通过)")
    for topic, ok, _ in results:
        if not ok:
            print(f"       ✗ {topic}")

    # gap 雷达 + idea 卡
    cards_ok = len(gc.GAPS) >= 5 and "最小实验" in gc.make_idea_card(gc.GAPS[0])
    print(f"  [{'OK ' if cards_ok else 'FAIL'}] gap 雷达 {len(gc.GAPS)} 个 + idea 卡生成")
    return assembly_ok and cards_ok


def main() -> int:
    print("== Part A: 依赖 ==")
    missing = check_imports()
    print("== Part B: nbformat ==")
    nb_ok = check_nbformat_roundtrip()
    print("== Part C: src 工具 (跨专题装配) ==")
    src_ok = check_src_tools()
    print("\n== 结论 ==")
    if missing:
        print(f"缺失依赖: {missing}\n  → pip install -r environment/requirements.txt")
        return 1
    if not (nb_ok and src_ok):
        print("检查未通过.")
        return 1
    print("全部通过 ✅  capstone 两个 notebook 可端到端运行 (含 M13 全链装配).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
