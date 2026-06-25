"""9.3 入口：生成点子 → 评委锦标赛排序（看 novelty）→ 对照真实执行排序，
现场看"评委冠军恰是执行垫底"和"评委偏爱自家点子"。

跑法：
  python learning/auto-research-frontier/m9.3-ideation-and-tournament/src/run.py
  python .../src/run.py --no-self-bias    # 关掉自偏好，看 novelty 偏差仍在
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from ideation import (                                  # noqa: E402
    IDEA_BANK, feasibility_of, judge_score, novelty, rank_by_feasibility, run_tournament,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="9.3 创意锦标赛：novelty vs feasibility")
    ap.add_argument("--no-self-bias", action="store_true",
                    help="关掉评委对自家点子的偏好（对比用）")
    ap.add_argument("--passes", type=int, default=8)
    args = ap.parse_args()

    use_bias = not args.no_self_bias
    ideas = list(IDEA_BANK)
    feas = feasibility_of(ideas)                         # 真实执行结果
    elo, judge_rank = run_tournament(ideas, use_self_bias=use_bias, passes=args.passes)
    feas_rank = rank_by_feasibility(ideas)
    feas_pos = {iid: i for i, iid in enumerate(feas_rank)}

    print(f"=== 9.3 创意锦标赛 | self_bias={'ON' if use_bias else 'OFF'} | {len(ideas)} 个点子 ===\n")
    print(f"  {'idea':12}{'src':>6}{'novelty':>9}{'judge':>8}{'elo':>8}"
          f"{'feasibility':>13}{'feas#':>7}")
    for iid in judge_rank:
        it = next(x for x in ideas if x.id == iid)
        print(f"  {iid:12}{it.source:>6}{novelty(it):>9.2f}"
              f"{judge_score(it, use_bias):>8.2f}{elo[iid]:>8.0f}"
              f"{feas[iid]:>13.3f}{feas_pos[iid] + 1:>7}")

    judge_top = judge_rank[0]
    feas_top = feas_rank[0]
    print(f"\n[教学点 1 · novelty ≠ feasibility]")
    print(f"  评委冠军 = '{judge_top}'（真实可行性排第 {feas_pos[judge_top] + 1}/{len(ideas)}，"
          f"acc={feas[judge_top]:.3f}）")
    print(f"  真实冠军 = '{feas_top}'（acc={feas[feas_top]:.3f}），评委把它排在第 "
          f"{judge_rank.index(feas_top) + 1} 位")
    if feas_pos[judge_top] >= len(ideas) // 2:
        print("  → 评委选出的'最佳点子'，真做出来其实在后半段。听着新 ≠ 做得出。")

    # —— 教学点 2：自偏好 ——
    self_ids = [it.id for it in ideas if it.source == "self"]
    other_ids = [it.id for it in ideas if it.source == "other"]
    m_elo_self = sum(elo[i] for i in self_ids) / len(self_ids)
    m_elo_other = sum(elo[i] for i in other_ids) / len(other_ids)
    m_feas_self = sum(feas[i] for i in self_ids) / len(self_ids)
    m_feas_other = sum(feas[i] for i in other_ids) / len(other_ids)
    print(f"\n[教学点 2 · 评委自偏好]")
    print(f"  self 点子：评委 Elo 均 {m_elo_self:.0f} | 真实 acc 均 {m_feas_self:.3f}")
    print(f"  other点子：评委 Elo 均 {m_elo_other:.0f} | 真实 acc 均 {m_feas_other:.3f}")
    if m_elo_self > m_elo_other and m_feas_self < m_feas_other:
        print("  → 自家点子真实更差，评委却给更高分。这就是 grading-own-homework 在 ideation 阶段的脸。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
