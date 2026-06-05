"""Arena-Hard (LMSYS 2024) — 500 hard prompts, pairwise judge.

Real protocol:
1. 500 questions sampled from Chatbot Arena that show high model-model
   variance
2. For each pair (model X, baseline GPT-4-0314), GPT-4-turbo judges
3. Win rate vs baseline = score
4. Bradley-Terry / Bradley-Terry-with-bootstrap for CI

Our toy: 8 MT-Bench prompts as the bench.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from common import GenFn, MT_BENCH_QS, PairBattle, PairJudgeFn, make_fixed_gen, make_length_judge


def run_pairwise(model_a: GenFn, name_a: str,
                  model_b: GenFn, name_b: str,
                  judge: PairJudgeFn) -> List[PairBattle]:
    out = []
    for s in MT_BENCH_QS:
        prompt = f"[qid={s.qid}] {s.prompt}"
        resp_a = model_a(prompt)
        resp_b = model_b(prompt)
        winner = judge(s.prompt, resp_a, resp_b)
        out.append(PairBattle(qid=s.qid, model_a=name_a,
                               model_b=name_b, winner=winner))
    return out


def win_rate(battles: List[PairBattle], model: str, treat_tie: str = "half") -> float:
    """Win rate of `model` (which is the 'A' side) over its opponent."""
    if not battles:
        return 0.0
    wins = sum(1 for b in battles if b.winner == "A")
    ties = sum(1 for b in battles if b.winner == "tie")
    n = len(battles)
    if treat_tie == "half":
        return (wins + 0.5 * ties) / n
    elif treat_tie == "drop":
        return wins / max(1, n - ties)
    elif treat_tie == "lose":
        return wins / n
    raise ValueError(treat_tie)


def _self_test() -> int:
    short = make_fixed_gen({s.qid: "short" for s in MT_BENCH_QS})
    long = make_fixed_gen({s.qid: "this is a much longer response than the other one"
                            for s in MT_BENCH_QS})
    judge = make_length_judge(prefer_longer=True)
    battles = run_pairwise(short, "short_model", long, "long_model", judge)
    wr_short = win_rate(battles, "short_model")
    assert wr_short == 0.0
    # Flip sides
    battles2 = run_pairwise(long, "long_model", short, "short_model", judge)
    assert win_rate(battles2, "long_model") == 1.0
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"arena_hard_runner.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
