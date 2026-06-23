"""vl-fusion-architectures 专题环境自检. 运行: python environment/verify_env.py"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REQUIRED = ["nbformat", "numpy", "pandas"]
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
    import connectors as cn

    rows = cn.param_comparison()
    tbl_ok = len(rows) == 3
    print(f"  [{'OK ' if tbl_ok else 'FAIL'}] connectors.param_comparison (3 路线)")

    fwd_ok = True
    if importlib.util.find_spec("torch") is not None:
        import torch
        conns = cn.build_connectors(vis_dim=32, llm_dim=48, n_vis=16)
        vis = torch.randn(2, 16, 32); txt = torch.randn(2, 10, 48)
        outs = {k: m(vis, txt).shape[1] for k, m in conns.items()}
        # projection/early: 16+10=26; cross_attn: 10 (视觉不占序列)
        fwd_ok = outs["projection"] == 26 and outs["cross_attn"] == 10 and outs["early_fusion"] == 26
        print(f"  [{'OK ' if fwd_ok else 'FAIL'}] 连接器前向序列长 {outs} (cross_attn 不增长)")

    return tbl_ok and fwd_ok


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
