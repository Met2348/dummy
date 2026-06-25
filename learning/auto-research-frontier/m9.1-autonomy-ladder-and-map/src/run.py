"""9.1 入口：把系统目录归到 Tool/Analyst/Scientist 三级，画全景地图，
并**程序化地**揭示一个不舒服的事实——2026 年自称"Scientist"的系统恰恰都是自评的。

跑法：
  python learning/auto-research-frontier/m9.1-autonomy-ladder-and-map/src/run.py
  python .../src/run.py --show table
  python .../src/run.py --system "AI Scientist v2"
"""
from __future__ import annotations

import argparse
import pathlib
import sys

# sys.path 引导：让 `python <repo>/.../src/run.py` 能 import 同目录的包
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from autonomy_map import (                       # noqa: E402
    LEVEL_RANK, SYSTEM_CATALOG, classify, classify_all, render_map, render_table,
)


def _insight(classifications) -> str:
    """从数据里算出"自主性自称 vs 可信度"的反相关，不硬编码结论。"""
    scientists = [c for c in classifications if c.evidenced_level == "scientist"]
    self_graded_sci = [c for c in scientists if c.self_verified_only]
    indep = [c for c in classifications if not c.self_verified_only]
    max_indep = max((LEVEL_RANK[c.evidenced_level] for c in indep), default=-1)
    levels = ("tool", "analyst", "scientist")
    max_indep_name = levels[max_indep] if max_indep >= 0 else "（无）"
    return (
        f"  · 证据级别为 Scientist 的系统：{len(scientists)} 个，其中 "
        f"{len(self_graded_sci)} 个**结果仅靠自评**（无独立验证）。\n"
        f"  · 反过来，经过独立验证的系统，证据级别最高只到 **{max_indep_name}**。\n"
        f"  → 2026 的拧巴现实：**自称越自主（Scientist），结果越是自己说了算；"
        f"真正可独立验证的，反而都还只是 Analyst。** 读这个领域，先把这两栏拆开看。"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="9.1 自主性阶梯 × 生命周期 全景地图")
    ap.add_argument("--show", default="all", choices=["map", "table", "all"])
    ap.add_argument("--system", default=None, help="只看某个系统的归类与理由")
    args = ap.parse_args()

    if args.system:
        match = [s for s in SYSTEM_CATALOG if s.name.lower() == args.system.lower()]
        if not match:
            print(f"未找到系统：{args.system}")
            print("可选：" + ", ".join(s.name for s in SYSTEM_CATALOG))
            return 2
        c = classify(match[0])
        print(f"=== {c.name} ===")
        print(f"  自称级别 : {c.claimed_level}")
        print(f"  证据级别 : {c.evidenced_level}   (hype_gap={c.hype_gap:+d})")
        print(f"  覆盖生命周期: {', '.join(c.coverage)}")
        print("  判定理由 :")
        for r in c.reasons:
            print(f"    - {r}")
        return 0

    classifications = classify_all()
    print(f"=== 9.1 自主性阶梯全景 | 共 {len(classifications)} 个系统 ===\n")
    if args.show in ("map", "all"):
        print(render_map(classifications))
        print()
    if args.show in ("table", "all"):
        print(render_table(classifications))
        print()
    print("[领域阅读法 · 从地图里读出的事实]")
    print(_insight(classifications))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
