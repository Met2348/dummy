"""circuits-attention 专题环境自检. 运行: python environment/verify_env.py"""
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
        print(f"  [{'OK ' if ok else 'opt '}] {pkg} (可选: 本专题需真实 gpt2)")
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
    sys.path.insert(0, str(src.parent.parent / "_shared"))
    import circuits as ci
    try:
        import realmodels as rm
        tok, model = rm.gpt2(output_attentions=True)
    except Exception:
        model = None

    if model is None:
        print("  [opt ] 无 gpt2 缓存/transformers, 跳过 induction 检查 (notebook 会优雅跳过)")
        return True
    import numpy as np
    tokens, k = ci.make_repeated_tokens(tok, n_unique=20, seed=0)
    scores = ci.induction_scores(model, tokens, k)
    best = np.unravel_index(np.argmax(scores), scores.shape)
    ind_ok = scores[best] > 0.4
    print(f"  [{'OK ' if ind_ok else 'FAIL'}] 找到 induction head (层{best[0]}头{best[1]} 分数 {scores[best]:.2f})")
    base = ci.induction_loss(model, tokens, k)
    abl = ci.induction_loss(model, tokens, k, ablate=tuple(best))
    abl_ok = abl >= base - 1e-3
    print(f"  [{'OK ' if abl_ok else 'FAIL'}] 消融 induction head (loss {base:.2f}→{abl:.2f})")
    return ind_ok and abl_ok


def main() -> int:
    print("== Part A: 依赖 ==")
    missing = check_imports()
    print("== Part B: nbformat ==")
    nb_ok = check_nbformat_roundtrip()
    print("== Part C: src 工具 (真实 gpt2 induction) ==")
    src_ok = check_src_tools()
    print("\n== 结论 ==")
    if missing:
        print(f"缺失依赖: {missing}\n  → pip install -r environment/requirements.txt")
        return 1
    if not (nb_ok and src_ok):
        print("检查未通过.")
        return 1
    print("全部通过 ✅  两个 notebook 可端到端运行 (需 gpt2 HF 缓存; 无则优雅跳过).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
