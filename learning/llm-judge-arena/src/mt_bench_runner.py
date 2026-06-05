"""MT-Bench (Zheng 2023) — 80 multi-turn questions.

Real protocol:
1. 80 prompts × 8 categories (writing/roleplay/reasoning/math/code/extract/STEM/humanities)
2. 2-turn: first prompt + follow-up
3. GPT-4 judge gives 1-10 score per turn
4. Model score = mean over questions/turns

Our toy: 8 1-turn prompts + 1-5 score judge.
"""
from __future__ import annotations

from typing import Dict, List

from common import GenFn, MT_BENCH_QS, PointJudgeFn, make_fixed_gen


def make_keyword_point_judge(good_kw: List[str]) -> PointJudgeFn:
    """Toy pointwise judge: +1 per good keyword, range 1-5."""
    def _fn(q: str, ans: str) -> int:
        c = sum(1 for k in good_kw if k.lower() in ans.lower())
        return max(1, min(5, 1 + c))
    return _fn


def run_mt_bench(gen: GenFn, judge: PointJudgeFn) -> Dict:
    """Score each question 1-5; return overall + per-category mean."""
    per_cat: Dict[str, List[int]] = {}
    per_q: Dict[str, int] = {}
    for s in MT_BENCH_QS:
        full_prompt = f"[qid={s.qid}] (Category: {s.category})\n{s.prompt}"
        ans = gen(full_prompt)
        sc = judge(s.prompt, ans)
        per_q[s.qid] = sc
        per_cat.setdefault(s.category or "_", []).append(sc)
    overall = sum(per_q.values()) / max(1, len(per_q))
    return {
        "overall": overall,
        "by_category": {k: sum(v)/len(v) for k, v in per_cat.items()},
        "per_q": per_q,
    }


def _self_test() -> int:
    j = make_keyword_point_judge(["good", "helpful", "step", "specifically", "Paris"])
    bad = make_fixed_gen({}, default="?")
    good = make_fixed_gen({s.qid: "This is helpful and step-by-step. Specifically very good."
                            for s in MT_BENCH_QS})
    rs_bad = run_mt_bench(bad, j)
    rs_good = run_mt_bench(good, j)
    assert rs_bad["overall"] < rs_good["overall"]
    assert rs_good["overall"] == 5.0
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"mt_bench_runner.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
