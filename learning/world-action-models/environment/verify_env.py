"""world-action-models 专题环境自检. 运行: python environment/verify_env.py"""
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
    import toy_env as env

    ok = True
    if importlib.util.find_spec("torch") is not None:
        S, A, D = wm.make_random_transitions(n=3000, seed=0)
        rand_ok = S.shape[1] == env.STATE_DIM and A.shape[1] == env.ACT_DIM
        print(f"  [{'OK ' if rand_ok else 'FAIL'}] 随机转移 (无专家) {S.shape}")
        model = wm.build_world_model()
        losses = wm.train_world_model(model, S, A, D, epochs=400)
        wm_ok = losses[-1] < losses[0]
        print(f"  [{'OK ' if wm_ok else 'FAIL'}] 世界模型训练 (loss {losses[0]:.4f}→{losses[-1]:.4f})")
        sr = env.eval_policy(wm.mpc_policy_fn(model, n_samples=150, horizon=6), n_episodes=60)
        mpc_ok = sr > 0.7
        print(f"  [{'OK ' if mpc_ok else 'FAIL'}] MPC 规划成功率 {sr:.2f} (>0.7, 零专家!)")
        ok = rand_ok and wm_ok and mpc_ok
    else:
        print("  [opt ] torch 缺失, 跳过世界模型检查")
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
    print("全部通过 ✅  两个 notebook 可端到端运行.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
