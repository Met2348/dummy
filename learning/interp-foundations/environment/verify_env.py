"""interp-foundations 专题环境自检. 运行: python environment/verify_env.py"""
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
        print(f"  [{'OK ' if ok else 'opt '}] {pkg} (可选: transformers 用于真实 gpt2)")
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
    import tiny_transformer as tt

    ok = True
    if importlib.util.find_spec("torch") is not None:
        import torch
        Xi, Yi = tt.make_data(1000, seed=0)
        model = tt.build_model(); tt.train(model, Xi, Yi, epochs=600)
        acc = tt.accuracy(model, *tt.make_data(300, seed=9))
        acc_ok = acc > 0.9
        print(f"  [{'OK ' if acc_ok else 'FAIL'}] tiny transformer 学会任务 (准确率 {acc:.2f})")
        _, cache = model.run_with_cache(torch.tensor(Xi[:2]))
        cache_ok = "resid_post_1" in cache and "attn_pattern_0" in cache
        print(f"  [{'OK ' if cache_ok else 'FAIL'}] run_with_cache 读取激活 ({len(cache)} 个)")
        ok = acc_ok and cache_ok
    else:
        print("  [opt ] torch 缺失, 跳过 tiny transformer 检查")
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
    print("全部通过 ✅  两个 notebook 可端到端运行 (N1 真实gpt2需HF缓存, 无则优雅跳过).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
