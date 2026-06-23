"""flow-matching-sota 专题环境自检. 运行: python environment/verify_env.py"""
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
    import flow_matching as fm

    x1 = fm.make_two_moons(n=256, seed=1)
    ok = True
    if importlib.util.find_spec("torch") is not None:
        m = fm.build_velocity_field()
        losses = fm.train_flow_matching(m, x1, epochs=300)
        train_ok = losses[-1] < losses[0]
        print(f"  [{'OK ' if train_ok else 'FAIL'}] flow matching 训练 (loss {losses[0]:.2f}→{losses[-1]:.2f})")
        # 步数越多质量越好
        q = fm.quality_vs_steps(m, x1, step_list=(2, 16))
        step_ok = q[1]["std_err"] <= q[0]["std_err"]
        print(f"  [{'OK ' if step_ok else 'FAIL'}] 步数↑质量↑ (2步 {q[0]['std_err']} ≥ 16步 {q[1]['std_err']})")
        # reflow 让少步更好
        m2, _ = fm.reflow(m, x1, epochs=300)
        before = fm.quality_vs_steps(m, x1, step_list=(4,))[0]["std_err"]
        after = fm.quality_vs_steps(m2, x1, step_list=(4,))[0]["std_err"]
        reflow_ok = after <= before
        print(f"  [{'OK ' if reflow_ok else 'FAIL'}] reflow 改善少步质量 (4步: {before}→{after})")
        ok = train_ok and step_ok and reflow_ok
    return ok


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
        print("检查未通过.")
        return 1
    print("全部通过 ✅  两个 notebook 可端到端运行.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
