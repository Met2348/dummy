"""9.6 入口：3 候选 × 3 rubric 的判分矩阵，看清弱 rubric 被刷、强 rubric 抗刷，
并验证沙箱真的拦得住越权候选。

跑法：
  python learning/auto-research-frontier/m9.6-evaluating-research-agents/src/run.py
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from mini_eval import (                                       # noqa: E402
    CANDIDATES, MALICIOUS_SRC, RUBRICS, SafeExecError, run_eval, safe_exec,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="9.6 评测 research agent：弱 rubric vs 强 rubric")
    ap.parse_args()

    rubric_names = list(RUBRICS)
    print("=== 9.6 判分矩阵（score / PASS|fail）| 行=候选实现 列=rubric ===\n")
    header = f"  {'candidate':14}" + "".join(f"{rn:>18}" for rn in rubric_names)
    print(header)
    matrix = {}
    for cname, src in CANDIDATES.items():
        cells = []
        for rn in rubric_names:
            r = run_eval(src, RUBRICS[rn])
            matrix[(cname, rn)] = r
            tag = "PASS" if r["passed"] else "fail"
            cells.append(f"{r['score']:.2f} {tag:>4}")
        print(f"  {cname:14}" + "".join(f"{c:>18}" for c in cells))

    print("\n[教学点 1 · 弱 rubric 被刷]")
    hv = matrix[("hardcode", "visible_only(弱)")]
    hh = matrix[("hardcode", "heldout(强)")]
    print(f"  硬编码候选：visible_only 判 {hv['score']:.2f}({'PASS' if hv['passed'] else 'fail'})"
          f" → 它只是背下了可见样本；")
    print(f"             heldout    判 {hh['score']:.2f}({'PASS' if hh['passed'] else 'fail'})"
          f" → 一上没见过的就露馅。")

    print("\n[教学点 2 · 信任自报指标的 rubric 是反的]")
    pf = matrix[("print-fraud", "trust_print(弱)")]
    ho = matrix[("honest", "trust_print(弱)")]
    print(f"  print 造假候选：trust_print 判 {pf['score']:.2f}({'PASS' if pf['passed'] else 'fail'})"
          f"（它只是 print 了个 1.0）；")
    print(f"  诚实候选：    trust_print 判 {ho['score']:.2f}({'PASS' if ho['passed'] else 'fail'})"
          f" → 不吹牛反而不过。信任自报 = 奖励说谎者。")

    print("\n[教学点 3 · 只有强 rubric(heldout) 区分出诚实]")
    for cname in CANDIDATES:
        r = matrix[(cname, "heldout(强)")]
        print(f"  {cname:14} heldout: {r['score']:.2f} {'PASS' if r['passed'] else 'fail'}")
    print("  → 在候选**没见过**的数据上真跑，是抗刷评测的最低门槛。")

    # —— 沙箱真的拦得住越权吗 ——
    print("\n[教学点 4 · 评测地基：沙箱真在限制]")
    try:
        safe_exec(MALICIOUS_SRC)
        print("  [X] 越权候选竟然跑通了——沙箱失效！")
    except SafeExecError as e:
        print(f"  越权候选（import os）被拦：{e}")
    print("  评测的第一性原理：先能**安全地真跑**候选，判分才有意义（接 agent-code-eval）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
