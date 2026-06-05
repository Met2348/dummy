"""AlpacaEval 2 LC (length-controlled, Dubois 2024).

Real protocol:
- 805 instructions from AlpacaFarm
- GPT-4-turbo judges pairwise (model vs baseline GPT-4-turbo)
- Length-controlled (LC) win rate: adjusts for verbosity bias via
  logistic regression that removes the effect of length difference

LC formula sketch:
  raw_winrate = mean(judge_says_model_wins)
  length_pred = sigmoid(α * (len_model - len_baseline))
  lc_winrate  = raw_winrate - length_pred + 0.5
"""
from __future__ import annotations

import math
from typing import Dict, List

from common import PairBattle


def lc_winrate(battles: List[PairBattle], lengths: Dict[str, int],
                alpha: float = 0.005) -> Dict[str, float]:
    """Compute raw + length-controlled win rate for model A.

    Args:
        battles: 'A' = focal model, 'B' = baseline
        lengths: {qid: len_A - len_B}
        alpha: logistic slope of length effect
    """
    if not battles:
        return {"raw_winrate": 0.0, "lc_winrate": 0.0}
    raw_wins = []
    length_effects = []
    for b in battles:
        win = 1.0 if b.winner == "A" else (0.5 if b.winner == "tie" else 0.0)
        raw_wins.append(win)
        dlen = lengths.get(b.qid, 0)
        length_effects.append(1.0 / (1.0 + math.exp(-alpha * dlen)))
    raw = sum(raw_wins) / len(raw_wins)
    le_mean = sum(length_effects) / len(length_effects)
    lc = raw - (le_mean - 0.5)
    return {"raw_winrate": raw, "lc_winrate": max(0.0, min(1.0, lc))}


def _self_test() -> int:
    # Equal length, A wins 60% → raw = lc = 0.6
    battles = [PairBattle(f"q{i}", "A", "B", "A" if i < 6 else "B")
               for i in range(10)]
    lengths = {f"q{i}": 0 for i in range(10)}
    r = lc_winrate(battles, lengths)
    assert abs(r["raw_winrate"] - 0.6) < 1e-9
    assert abs(r["lc_winrate"] - 0.6) < 1e-9
    # A is much longer: lc should pull below raw (length-adjusted)
    lengths_long = {f"q{i}": 500 for i in range(10)}
    r2 = lc_winrate(battles, lengths_long)
    assert r2["lc_winrate"] < r2["raw_winrate"], r2
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"alpaca_eval.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
