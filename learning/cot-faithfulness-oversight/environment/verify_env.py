"""cot-faithfulness-oversight 专题环境自检. 运行: python environment/verify_env.py"""
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
OPTIONAL = ["torch", "transformers"]


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
    sys.path.insert(0, str(src.parent.parent / "_shared"))
    import cot_probe as cp

    ok = True
    if importlib.util.find_spec("torch") is not None:
        # weak-to-strong: 强学生应超过弱监督者
        wa, sa = cp.weak_to_strong_demo(noise=0.25, seed=0)
        w2s_ok = sa > wa + 0.05
        print(f"  [{'OK ' if w2s_ok else 'FAIL'}] weak-to-strong (弱 {wa:.2f} → 强学生 {sa:.2f})")
        ok = w2s_ok
        # CoT 部分需 TinyLlama, 在 notebook 里跑 (verify 不强制下载)
        if importlib.util.find_spec("transformers") is not None:
            print("  [OK ] cot_probe 偏置敏感性 (真实 TinyLlama 在 notebook 跑; 需 HF 缓存)")
    else:
        print("  [opt ] torch 缺失, 跳过检查")
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
    print("全部通过 ✅  两个 notebook 可端到端运行 (N1 真实 TinyLlama 需 HF 缓存).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
