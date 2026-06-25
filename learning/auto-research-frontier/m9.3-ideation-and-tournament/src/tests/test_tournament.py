"""V2 测试：锁死 9.3 的诚实性——真实训练确定性、评委冠军恰是执行垫底、
自偏好可测、关掉自偏好后 novelty 偏差仍在。
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ideation import (
    IDEA_BANK, Idea, feasibility_of, judge_score, novelty,
    rank_by_feasibility, run_tournament, train_logreg,
)


def test_training_is_real_and_deterministic():
    a = train_logreg({"steps": 500})
    b = train_logreg({"steps": 500})
    assert a == b
    # 真随 config 变：多训应优于极少步数
    assert train_logreg({"steps": 2000}) > train_logreg({"steps": 20})


def test_self_preference_is_measurable():
    """同样文案，source=self 比 other 评分更高——纯粹的自偏好。"""
    text = "一个普通的小改动。"
    s = Idea("x-self", text, {}, source="self")
    o = Idea("x-other", text, {}, source="other")
    assert judge_score(s) > judge_score(o)
    assert judge_score(s, use_self_bias=False) == judge_score(o, use_self_bias=False)


def test_judge_champion_is_low_feasibility():
    """核心：评委冠军的真实可行性排在后半段（novelty≠feasibility）。"""
    ideas = list(IDEA_BANK)
    _, judge_rank = run_tournament(ideas, use_self_bias=True)
    feas_rank = rank_by_feasibility(ideas)
    champ = judge_rank[0]
    pos = feas_rank.index(champ)             # 0=最佳
    assert pos >= len(ideas) // 2, f"评委冠军 {champ} 真实排名 {pos+1}，本应靠后"


def test_real_champion_is_underrated_by_judge():
    """真实最佳点子被评委低估（排不到前列）。"""
    ideas = list(IDEA_BANK)
    _, judge_rank = run_tournament(ideas, use_self_bias=True)
    feas_rank = rank_by_feasibility(ideas)
    real_best = feas_rank[0]
    assert judge_rank.index(real_best) >= 2, "真实冠军不该被评委排进前二"


def test_self_ideas_worse_but_judged_better():
    """self 点子真实平均更差，Elo 平均却更高——执行 vs 评分的反转。"""
    ideas = list(IDEA_BANK)
    elo, _ = run_tournament(ideas, use_self_bias=True)
    feas = feasibility_of(ideas)
    self_ids = [it.id for it in ideas if it.source == "self"]
    other_ids = [it.id for it in ideas if it.source == "other"]
    mean = lambda d, ks: sum(d[k] for k in ks) / len(ks)
    assert mean(feas, self_ids) < mean(feas, other_ids)     # 真实更差
    assert mean(elo, self_ids) > mean(elo, other_ids)       # 评分更高


def test_novelty_bias_survives_without_self_bias():
    """即便关掉自偏好，高 novelty 文案仍把低可行性点子顶上去——偏差不止自偏好一种。"""
    ideas = list(IDEA_BANK)
    _, judge_rank = run_tournament(ideas, use_self_bias=False)
    feas_rank = rank_by_feasibility(ideas)
    assert feas_rank.index(judge_rank[0]) >= len(ideas) // 2


if __name__ == "__main__":   # 直跑兜底
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                fails += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"\n{'OK' if fails == 0 else f'{fails} FAILED'}")
    raise SystemExit(1 if fails else 0)
