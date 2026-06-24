"""action-heads-diffusion-policy 专题环境自检. 运行: python environment/verify_env.py"""
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
    import numpy as np
    import diffusion_policy as dp

    S, A = dp.make_obstacle_demos(n=200, chunk=1, seed=0)
    data_ok = S.shape[1] == dp.STATE_DIM and A.shape[1] == dp.ACT_DIM
    print(f"  [{'OK ' if data_ok else 'FAIL'}] 双峰 demo {S.shape}")

    ok = data_ok
    if importlib.util.find_spec("torch") is not None:
        import torch
        diff = dp.build_diffusion_policy(chunk=1)
        _, sched = dp.train_diffusion_policy(diff, S, A, epochs=500)
        reg = dp.build_regression(chunk=1); dp.train_regression(reg, S, A, epochs=500)
        # 多峰: 起点采样应跨越 y>0 和 y<0 两峰; 回归应塌成 |y| 小
        st = np.array([dp.START_X, 0.0], np.float32)
        acts = np.array([dp.sample_action(diff, st, sched, seed=k)[0] for k in range(80)])
        bimodal = (acts[:, 1] > 0.15).any() and (acts[:, 1] < -0.15).any()
        reg_a = reg(torch.tensor(st[None])).detach().numpy()[0]
        reg_avg = abs(reg_a[1]) < 0.3
        print(f"  [{'OK ' if bimodal else 'FAIL'}] 扩散动作头双峰 (采样跨 y>0 和 y<0)")
        print(f"  [{'OK ' if reg_avg else 'FAIL'}] 回归取平均 (动作 dy={reg_a[1]:+.2f} ≈0)")
        ok = data_ok and bimodal and reg_avg
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
