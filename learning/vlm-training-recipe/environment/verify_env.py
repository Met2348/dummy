"""vlm-training-recipe 专题环境自检. 运行: python environment/verify_env.py"""
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
        print(f"  [{'OK ' if ok else 'opt '}] {pkg} (可选; 无则训练部分跳过)")
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
    import mini_vlm as mv

    X, y, n_patch = mv.make_vqa_dataset(n_classes=4, n_per_class=12, seed=1)
    data_ok = X.shape[0] == 48 and len(set(y)) == 4
    print(f"  [{'OK ' if data_ok else 'FAIL'}] make_vqa_dataset ({X.shape[0]} 图 4 类)")

    train_ok = True
    if importlib.util.find_spec("torch") is not None:
        m = mv.build_mini_vlm(patch_dim=X.shape[2], n_patch=n_patch, n_classes=4)
        mv.set_freeze(m, freeze_vision=False, freeze_llm=False)
        loss, acc = mv.train_mini_vlm(m, X, y, epochs=30)
        train_ok = loss[-1] < loss[0] and acc[-1] > 0.7   # 学到了东西
        print(f"  [{'OK ' if train_ok else 'FAIL'}] mini-VLM 训练 (loss {loss[0]:.2f}→{loss[-1]:.2f}, acc→{acc[-1]:.2f})")
        # 冻结减少可训练参数
        mv.set_freeze(m, freeze_vision=True, freeze_llm=True)
        frozen_p = mv.count_trainable(m)
        mv.set_freeze(m, freeze_vision=False, freeze_llm=False)
        full_p = mv.count_trainable(m)
        freeze_ok = frozen_p < full_p
        print(f"  [{'OK ' if freeze_ok else 'FAIL'}] set_freeze (冻={frozen_p} < 全={full_p})")
        train_ok = train_ok and freeze_ok

    return data_ok and train_ok


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
