"""dit-latent-diffusion 专题环境自检. 运行: python environment/verify_env.py"""
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
    import dit

    x, y = dit.make_class_blobs(n_per=120, seed=1)
    data_ok = x.shape[0] == 480 and set(y.tolist()) == {0, 1, 2, 3}
    print(f"  [{'OK ' if data_ok else 'FAIL'}] make_class_blobs ({x.shape[0]} 点 4 类)")

    ok = data_ok
    if importlib.util.find_spec("torch") is not None:
        model = dit.build_dit()
        losses, sched = dit.train_dit(model, x, y, epochs=400)
        train_ok = losses[-1] < losses[0]
        print(f"  [{'OK ' if train_ok else 'FAIL'}] DiT 训练 (loss {losses[0]:.2f}→{losses[-1]:.2f})")
        # CFG: 有条件 (g=1) 类别准确率 > 无条件 (g=0)
        acc_cond = dit.class_accuracy(dit.sample(model, sched, cls=0, n=150, guidance=1.0, seed=2), 0)
        acc_unc = dit.class_accuracy(dit.sample(model, sched, cls=0, n=150, guidance=0.0, seed=2), 0)
        cfg_ok = acc_cond > acc_unc
        print(f"  [{'OK ' if cfg_ok else 'FAIL'}] CFG 条件生效 (g=1 acc {acc_cond:.2f} > g=0 acc {acc_unc:.2f})")
        ok = data_ok and train_ok and cfg_ok
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
