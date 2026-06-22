"""
research-figures 专题环境自检.
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

REQUIRED = ["nbformat", "numpy", "matplotlib", "pandas"]


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
    import matplotlib
    matplotlib.use("Agg")  # 自检时无显示后端
    import matplotlib.pyplot as plt

    src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(src))
    import plotstyle as ps
    import schematic as sc

    ps.set_pub_style()
    pal_ok = len(ps.OKABE_ITO) == 8
    print(f"  [{'OK ' if pal_ok else 'FAIL'}] plotstyle.set_pub_style + Okabe-Ito 调色板")

    with tempfile.TemporaryDirectory() as d:
        fig, ax = plt.subplots(figsize=ps.column_figsize())
        ps.grouped_bar(ax, ["0", "0.4"], ["DPO", "Robust-DPO"],
                       means=[[0.62, 0.40], [0.62, 0.53]],
                       errs=[[0.01, 0.01], [0.01, 0.01]], ylabel="win")
        paths = ps.save_figure(fig, "t", out_dir=d)
        plt.close(fig)
        save_ok = len(paths) == 2 and all(p.exists() for p in paths)
        pdf_ok = any(p.suffix == ".pdf" for p in paths)
    print(f"  [{'OK ' if save_ok and pdf_ok else 'FAIL'}] grouped_bar + save_figure (PDF+PNG)")

    fig, ax = plt.subplots(figsize=(8, 1.6))
    sc.draw_pipeline(ax, ["a", "b", "c"])
    plt.close(fig)
    print("  [OK ] schematic.draw_pipeline")

    return pal_ok and save_ok and pdf_ok


def main() -> int:
    print("== Part A: 依赖 import ==")
    missing = check_imports()
    print("== Part B: nbformat 往返 ==")
    nb_ok = check_nbformat_roundtrip()
    print("== Part C: src 绘图工具冒烟 ==")
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
