"""vlm-eval-hallucination 专题环境自检. 运行: python environment/verify_env.py"""
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
    src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(src))
    import vlm_eval as ve

    # 幻觉率↑ → 准确率↓, yes-rate↑ (单调)
    low = ve.run_pope(hallucination=0.0, n=400, seed=1)
    high = ve.run_pope(hallucination=0.8, n=400, seed=1)
    mono_ok = high["accuracy"] < low["accuracy"] and high["yes_rate"] > low["yes_rate"]
    print(f"  [{'OK ' if mono_ok else 'FAIL'}] POPE: 幻觉率↑→准确率↓({low['accuracy']}→{high['accuracy']}) yes率↑({low['yes_rate']}→{high['yes_rate']})")

    sig = ve.yes_bias_signal(high)
    sig_ok = "yes-bias" in sig and "幻觉" in sig
    print(f"  [{'OK ' if sig_ok else 'FAIL'}] yes_bias_signal 检出强幻觉")

    # 平衡探测集
    items = ve.make_probe_set(n=100, seed=0)
    bal = sum(it["truly_present"] for it in items)
    bal_ok = bal == 50
    print(f"  [{'OK ' if bal_ok else 'FAIL'}] make_probe_set 正负平衡 ({bal}/100 正)")

    return mono_ok and sig_ok and bal_ok


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
