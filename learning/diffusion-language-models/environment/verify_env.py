"""diffusion-language-models 专题环境自检. 运行: python environment/verify_env.py"""
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
    import diffusion_lm as dl

    data = dl.make_sequences(800, seed=0)
    data_ok = data.shape == (800, dl.L) and dl.is_palindrome(data) == 1.0
    print(f"  [{'OK ' if data_ok else 'FAIL'}] make_sequences {data.shape} 全回文")

    ok = data_ok
    if importlib.util.find_spec("torch") is not None:
        import torch
        torch.manual_seed(0)
        dlm = dl.build_dlm(d_model=80); dl.train_dlm(dlm, data, epochs=500)
        ar = dl.build_ar(d_model=80); dl.train_ar(ar, data, epochs=500)
        # 顺序解码合法率应高
        gen = dl.generate_dlm(dlm, n=200, rounds=dl.L, seed=1)
        gen_ok = dl.is_palindrome(gen) > 0.85
        print(f"  [{'OK ' if gen_ok else 'FAIL'}] dLLM 顺序解码合法率 {dl.is_palindrome(gen):.2f} (>0.85)")
        # 双向 infilling: dLLM 远胜 AR
        test = dl.make_sequences(200, seed=9)
        a_dlm = dl.dlm_infill_accuracy(dlm, test, 1)
        a_ar = dl.ar_infill_accuracy(ar, test, 1)
        infill_ok = a_dlm > 0.85 and a_ar < 0.4
        print(f"  [{'OK ' if infill_ok else 'FAIL'}] 双向 infilling: dLLM {a_dlm:.2f} >> AR {a_ar:.2f}")
        ok = data_ok and gen_ok and infill_ok
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
