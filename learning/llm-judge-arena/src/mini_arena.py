"""Capstone — mini Chatbot Arena with 5 mock models.

Runs round-robin pairwise battles, fits Bradley-Terry, reports Elo.
"""
from __future__ import annotations

import itertools
from typing import Dict, List

from common import GenFn, MT_BENCH_QS, PairBattle, PairJudgeFn, make_fixed_gen, make_length_judge
from arena_hard_runner import run_pairwise, win_rate
from bradley_terry import fit_bt, to_elo


# 5 mock models with intentional skill ordering by length verbosity
def make_arena_models() -> Dict[str, GenFn]:
    base_responses = {
        "vanilla":  {s.qid: "ok" for s in MT_BENCH_QS},
        "lora":     {s.qid: "ok, more detail here." for s in MT_BENCH_QS},
        "dpo":      {s.qid: "ok, more detail here, and a clear example for the user." for s in MT_BENCH_QS},
        "r1_tiny":  {s.qid: "ok, with a step-by-step reasoning chain. First, ... Second, ... Therefore the answer is shown above with thorough justification." for s in MT_BENCH_QS},
        "phi_tiny": {s.qid: "concise, accurate answer with one well-structured paragraph and explanation that is well-organized." for s in MT_BENCH_QS},
    }
    return {name: make_fixed_gen(resp) for name, resp in base_responses.items()}


def run_round_robin(models: Dict[str, GenFn], judge: PairJudgeFn) -> List[PairBattle]:
    out = []
    for a, b in itertools.combinations(models.keys(), 2):
        out.extend(run_pairwise(models[a], a, models[b], b, judge))
        out.extend(run_pairwise(models[b], b, models[a], a, judge))  # both orderings
    return out


def make_leaderboard(battles: List[PairBattle]) -> Dict:
    bt = fit_bt(battles, n_iter=200)
    elo = to_elo(bt)
    # rank by elo
    ranked = sorted(elo.items(), key=lambda kv: -kv[1])
    return {"bt_log_strength": bt, "elo": elo, "ranking": ranked}


def to_md(lb: Dict) -> str:
    lines = ["# mini-Arena leaderboard", "",
             "| rank | model | Elo |", "|---|---|---:|"]
    for i, (name, elo) in enumerate(lb["ranking"], start=1):
        lines.append(f"| {i} | {name} | {elo} |")
    return "\n".join(lines)


def _self_test() -> int:
    models = make_arena_models()
    judge = make_length_judge(prefer_longer=True)
    battles = run_round_robin(models, judge)
    # 5 choose 2 = 10 pairs × 2 orderings × 8 questions = 160
    assert len(battles) == 5 * 4 * len(MT_BENCH_QS)
    lb = make_leaderboard(battles)
    elo = lb["elo"]
    # Longest response = r1_tiny → highest Elo
    assert elo["r1_tiny"] >= elo["dpo"] >= elo["lora"] >= elo["vanilla"], elo
    md = to_md(lb)
    assert "r1_tiny" in md and "Elo" in md
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"mini_arena.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
    models = make_arena_models()
    judge = make_length_judge(prefer_longer=True)
    battles = run_round_robin(models, judge)
    print(to_md(make_leaderboard(battles)))
