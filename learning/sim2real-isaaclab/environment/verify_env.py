"""sim2real-isaaclab 专题环境自检. 运行: python environment/verify_env.py"""
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
    import domain_rand as dr

    notes_ok = (src / "isaaclab_notes.md").exists()
    print(f"  [{'OK ' if notes_ok else 'FAIL'}] isaaclab_notes.md (踩坑 checklist) 存在")

    ok = notes_ok
    if importlib.util.find_spec("torch") is not None:
        # DR 应弥合 gap: DR 在 real(全) 上优于无 DR
        S0, A0 = dr.collect_demos(n=300, region="narrow", seed=0)
        m0 = dr.build_policy(seed=0); dr.train_policy(m0, S0, A0)
        Sd, Ad = dr.collect_demos(n=300, region="wide", seed=0)
        md = dr.build_policy(seed=0); dr.train_policy(md, Sd, Ad)
        real0 = dr.eval_region(m0, "wide", n_episodes=150)
        reald = dr.eval_region(md, "wide", n_episodes=150)
        dr_ok = reald > real0 + 0.1
        print(f"  [{'OK ' if dr_ok else 'FAIL'}] DR 弥合 gap (real: 无DR {real0:.2f} < DR {reald:.2f})")
        ok = notes_ok and dr_ok
    return ok


def main() -> int:
    print("== Part A: 依赖 ==")
    missing = check_imports()
    print("== Part B: nbformat ==")
    nb_ok = check_nbformat_roundtrip()
    print("== Part C: src 工具 (含 M11.1 共享环境复用) ==")
    src_ok = check_src_tools()
    print("\n== 结论 ==")
    if missing:
        print(f"缺失依赖: {missing}\n  → pip install -r environment/requirements.txt")
        return 1
    if not (nb_ok and src_ok):
        print("检查未通过.")
        return 1
    print("全部通过 ✅  两个 notebook 可端到端运行 (N2 为指引, 不需 GPU).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
