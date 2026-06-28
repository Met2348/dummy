"""probing-and-activations 专题环境自检. 运行: python environment/verify_env.py"""
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
    import probing as pr
    import tiny_transformer as tt

    ok = True
    if importlib.util.find_spec("torch") is not None:
        import torch
        Xi, Yi = tt.make_data(1500, seed=0)
        model = tt.build_model(); tt.train(model, Xi, Yi, epochs=700)
        acts, labels = pr.tiny_layer_activations(model, tt.make_data(400, seed=5)[0], layer_key="resid_post_1")
        tr, te = pr.linear_probe(acts, labels, n_classes=tt.V)
        probe_ok = te > 0.9
        print(f"  [{'OK ' if probe_ok else 'FAIL'}] 线性探针读出'当前值' (准确率 {te:.2f})")
        lens = pr.logit_lens_tiny(model, tt.make_data(1, seed=7)[0])
        sample = tt.make_data(1, seed=7)[0]
        lens_ok = int(lens["resid_post_1"][0]) == int((sample[0, -1] + 1) % tt.V)
        print(f"  [{'OK ' if lens_ok else 'FAIL'}] logit lens 末层预测正确")
        ok = probe_ok and lens_ok
    else:
        print("  [opt ] torch 缺失, 跳过探针检查")
    return ok


def main() -> int:
    print("== Part A: 依赖 ==")
    missing = check_imports()
    print("== Part B: nbformat ==")
    nb_ok = check_nbformat_roundtrip()
    print("== Part C: src 工具 (复用 12.1 tiny_transformer) ==")
    src_ok = check_src_tools()
    print("\n== 结论 ==")
    if missing:
        print(f"缺失依赖: {missing}\n  → pip install -r environment/requirements.txt")
        return 1
    if not (nb_ok and src_ok):
        print("检查未通过.")
        return 1
    print("全部通过 ✅  两个 notebook 可端到端运行 (真实 gpt2 部分需 HF 缓存).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
