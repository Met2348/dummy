"""diffusion-foundations 专题环境自检. 运行: python environment/verify_env.py"""
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
        print(f"  [{'OK ' if ok else 'opt '}] {pkg} (可选; 无则训练/采样跳过)")
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
    import diffusion as df

    x0 = df.make_two_moons(n=256, seed=1)
    data_ok = x0.shape == (256, 2)
    print(f"  [{'OK ' if data_ok else 'FAIL'}] make_two_moons {x0.shape}")

    betas, alphas, abars = df.make_beta_schedule(T=50)
    sched_ok = abars[0] > abars[-1] and abars[-1] < 0.5   # 累积保留量递减到接近0
    print(f"  [{'OK ' if sched_ok else 'FAIL'}] beta schedule (ᾱ: {abars[0]:.2f}→{abars[-1]:.3f})")

    train_ok = True
    if importlib.util.find_spec("torch") is not None:
        model = df.build_denoiser()
        losses, sched = df.train_diffusion(model, x0, T=50, epochs=200)
        gen = df.sample(model, sched, n=256, seed=2)
        # 生成样本统计接近目标 (std 不崩)
        std_close = abs(gen.std() - x0.std()) < 0.4
        train_ok = losses[-1] < losses[0] and std_close
        print(f"  [{'OK ' if train_ok else 'FAIL'}] DDPM 训练+采样 (loss {losses[0]:.2f}→{losses[-1]:.2f}, gen std {gen.std():.2f})")

    return data_ok and sched_ok and train_ok


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
