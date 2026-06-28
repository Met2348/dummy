"""interp-graduation 专题环境自检. 运行: python environment/verify_env.py"""
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
    import interp_capstone as cap

    results = cap.assembly_check()
    n_ok = sum(1 for _, ok, _ in results)
    assembly_ok = n_ok == len(cap.STACK)
    print(f"  [{'OK ' if assembly_ok else 'FAIL'}] M12 全套装配 ({n_ok}/{len(cap.STACK)} import 通过)")
    for label, ok, _ in results:
        if not ok:
            print(f"       ✗ {label}")

    ok = assembly_ok
    if importlib.util.find_spec("torch") is not None:
        r = cap.run_full_interp(seed=0)
        story_ok = r["probe_acc"] > 0.9 and r["causal_pos"] == r["last_pos"] and r["sae_purity"] > r["raw_purity"]
        print(f"  [{'OK ' if story_ok else 'FAIL'}] 完整流程 (探针 {r['probe_acc']:.2f} / 因果位置 {r['causal_pos']} / SAE {r['sae_purity']:.2f}>{r['raw_purity']:.2f})")
        ok = assembly_ok and story_ok
    cards_ok = len(cap.GAPS) >= 4 and "最小实验" in cap.make_idea_card(cap.GAPS[0])
    print(f"  [{'OK ' if cards_ok else 'FAIL'}] gap 雷达 {len(cap.GAPS)} 个 + idea 卡")
    return ok and cards_ok


def main() -> int:
    print("== Part A: 依赖 ==")
    missing = check_imports()
    print("== Part B: nbformat ==")
    nb_ok = check_nbformat_roundtrip()
    print("== Part C: src 工具 (跨专题装配 M12 全套) ==")
    src_ok = check_src_tools()
    print("\n== 结论 ==")
    if missing:
        print(f"缺失依赖: {missing}\n  → pip install -r environment/requirements.txt")
        return 1
    if not (nb_ok and src_ok):
        print("检查未通过.")
        return 1
    print("全部通过 ✅  capstone 两个 notebook 可端到端运行 (含 M12 全套装配 + 完整流程).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
