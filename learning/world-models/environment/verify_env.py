"""world-models 专题环境自检. 运行: python environment/verify_env.py"""
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
    import world_model as wm

    data = wm.make_transitions(n=500, seed=0)
    data_ok = data[0].shape == (500, 2) and data[2].shape == (500, 2)
    print(f"  [{'OK ' if data_ok else 'FAIL'}] make_transitions {data[0].shape}")

    ok = data_ok
    if importlib.util.find_spec("torch") is not None:
        model = wm.build_world_model()
        losses = wm.train_world_model(model, data, epochs=200)
        train_ok = losses[-1] < losses[0]
        print(f"  [{'OK ' if train_ok else 'FAIL'}] 世界模型训练 (loss {losses[0]:.4f}→{losses[-1]:.4f})")
        # 误差应随步数增长 (累积)
        errs = wm.multistep_error(model, n_traj=30, horizon=15)
        grow_ok = errs[15] > errs[1]
        print(f"  [{'OK ' if grow_ok else 'FAIL'}] 多步误差累积 (1步 {errs[1]:.3f} < 15步 {errs[15]:.3f})")
        ok = data_ok and train_ok and grow_ok
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
