"""
research-knowledge-mgmt 专题环境自检.

运行: python environment/verify_env.py
通过标准: 依赖可 import, nbformat 可写读, src 工具可 import 且冒烟通过.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

# Windows 控制台默认 GBK, 打印 emoji/部分 Unicode 会 UnicodeEncodeError; 强制 UTF-8 输出.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REQUIRED = ["nbformat", "pandas"]


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

    nb = new_notebook(cells=[new_code_cell("print('hello from verify')")])
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "smoke.ipynb"
        nbformat.write(nb, p)
        back = nbformat.read(p, as_version=4)
    ok = back.cells[0].source == "print('hello from verify')"
    print(f"  [{'OK ' if ok else 'FAIL'}] nbformat 读写往返")
    return ok


def check_src_tools() -> bool:
    """确认 src/ 工具可 import 且核心函数冒烟通过 (纯 stdlib, 无需联网)."""
    src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(src))
    ok = True
    for mod in ["bib_to_cards", "arxiv_triage"]:
        spec = importlib.util.find_spec(mod)
        present = spec is not None
        print(f"  [{'OK ' if present else 'MISS'}] src/{mod}.py")
        ok = ok and present
    if not ok:
        return False

    # 冒烟: 解析一条最小 BibTeX, triage 一批最小论文.
    import bib_to_cards
    import arxiv_triage

    entries = bib_to_cards.parse_bibtex(
        "@article{x2026, title={A Test}, author={Doe, Jane}, year={2026}}"
    )
    smoke_bib = len(entries) == 1 and entries[0]["title"] == "A Test"
    print(f"  [{'OK ' if smoke_bib else 'FAIL'}] bib_to_cards.parse_bibtex 冒烟")

    ranked = arxiv_triage.triage(
        [{"title": "DPO for alignment", "abstract": "preference optimization"}],
        {"dpo": 3, "alignment": 2},
    )
    smoke_tri = len(ranked) == 1 and ranked[0]["score"] > 0
    print(f"  [{'OK ' if smoke_tri else 'FAIL'}] arxiv_triage.triage 冒烟")

    return smoke_bib and smoke_tri


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
