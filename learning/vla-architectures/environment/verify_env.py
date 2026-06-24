"""vla-architectures 专题环境自检. 运行: python environment/verify_env.py"""
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
    import mini_vla as vla   # 内部会把 M11.1 的 toy_env 加进 path
    import toy_env as env

    ok = True
    if importlib.util.find_spec("torch") is not None:
        S, A = env.make_demos(n=200, seed=0)
        for head in ["discrete", "continuous"]:
            m = vla.build_mini_vla(head=head)
            vla.train_vla(m, S, A, epochs=300)
            sr = env.eval_policy(vla.make_policy(m), n_episodes=100)
            head_ok = sr > 0.9
            print(f"  [{'OK ' if head_ok else 'FAIL'}] mini-VLA {head} 头成功率 {sr:.2f} (>0.9)")
            ok = ok and head_ok
        # 连续头应比离散头平滑
        md = vla.build_mini_vla(head="discrete"); vla.train_vla(md, S, A, epochs=300)
        mc = vla.build_mini_vla(head="continuous"); vla.train_vla(mc, S, A, epochs=300)
        smooth_ok = vla.action_smoothness(mc) < vla.action_smoothness(md)
        print(f"  [{'OK ' if smooth_ok else 'FAIL'}] 连续头比离散头平滑 "
              f"({vla.action_smoothness(mc):.3f} < {vla.action_smoothness(md):.3f})")
        ok = ok and smooth_ok
    else:
        print("  [opt ] torch 缺失, 跳过 VLA 训练检查")
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
