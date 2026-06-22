"""
paper-writing-submission 专题环境自检.
运行: python environment/verify_env.py
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REQUIRED = ["nbformat"]


def check_imports() -> list[str]:
    missing = []
    for pkg in REQUIRED:
        ok = importlib.util.find_spec(pkg) is not None
        print(f"  [{'OK ' if ok else 'MISS'}] {pkg}")
        if not ok:
            missing.append(pkg)
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
    import paper_assembler as pa
    import rebuttal_kit as rk

    arts = {"narrative": "x", "results": "y", "ablation": "z"}
    claims = [{"claim": "A", "evidence": "results"}, {"claim": "B", "evidence": None}]
    rep = pa.narrative_audit(arts, claims)
    pa_ok = rep["sections_covered"] > 0 and len(rep["unsupported_claims"]) == 1
    print(f"  [{'OK ' if pa_ok else 'FAIL'}] paper_assembler.narrative_audit (抓出 1 个无证据 claim)")

    comments = ["You should add a baseline experiment.", "This is interesting.",
                "The notation is unclear."]
    tri = rk.triage(comments)
    # 补实验应排在正面评价前
    rk_ok = tri[0]["category"] == "add_experiment" and tri[-1]["category"] == "positive"
    print(f"  [{'OK ' if rk_ok else 'FAIL'}] rebuttal_kit.triage (补实验优先, 正面垫后)")

    skel = rk.build_skeleton(comments)
    skel_ok = "Rebuttal" in skel and "词" in skel
    print(f"  [{'OK ' if skel_ok else 'FAIL'}] rebuttal_kit.build_skeleton")

    return pa_ok and rk_ok and skel_ok


def main() -> int:
    print("== Part A: 依赖 import ==")
    missing = check_imports()
    print("== Part B: nbformat 往返 ==")
    nb_ok = check_nbformat_roundtrip()
    print("== Part C: src 工具冒烟 ==")
    src_ok = check_src_tools()

    print("\n== 结论 ==")
    if missing:
        print(f"缺失依赖: {missing}\n  → pip install -r environment/requirements.txt")
        return 1
    if not (nb_ok and src_ok):
        print("nbformat 或 src 工具检查未通过.")
        return 1
    print("全部通过 ✅  两个 notebook 可端到端运行.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
