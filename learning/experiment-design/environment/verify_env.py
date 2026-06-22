"""
experiment-design 专题环境自检.
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
OPTIONAL = ["scipy"]  # 没有也能跑 (stats.py 回退正态近似)


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
    import experiment as ex
    import stats as st

    # 确定性: 同一调用两次必须一致 (这正是本专题 L5 强调的可复现)
    a = ex.run_experiment("DPO", 0.4, 3)
    b = ex.run_experiment("DPO", 0.4, 3)
    det = (a == b)
    print(f"  [{'OK ' if det else 'FAIL'}] experiment.run_experiment 确定性 ({a:.4f})")

    runs = ex.ablation_grid(seeds=range(5))
    grid_ok = len(runs) == 2 * 3 * 5
    print(f"  [{'OK ' if grid_ok else 'FAIL'}] experiment.ablation_grid ({len(runs)} runs)")

    # 埋的交互效应: 高噪声下 Robust 应显著高于 DPO
    rob = ex.runs_for(runs, method="Robust-DPO", noise=0.4)
    dpo = ex.runs_for(runs, method="DPO", noise=0.4)
    t, p = st.welch_t_test(rob, dpo)
    inter_ok = (sum(rob) / len(rob) > sum(dpo) / len(dpo)) and p < 0.05
    print(f"  [{'OK ' if inter_ok else 'FAIL'}] 交互效应可检出 (noise=0.4: p={p:.4f})")

    return det and grid_ok and inter_ok


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
