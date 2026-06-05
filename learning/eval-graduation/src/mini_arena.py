"""Capstone-2 part A: mini-Arena — 5 ckpt round-robin + BT-Elo."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ckpt_zoo import CkptInterface, load_all, KEY_QUESTIONS


@dataclass
class Battle:
    qkey: str
    a: str
    b: str
    winner: str  # 'A' / 'B' / 'tie'


def _length_quality_judge(r_a: str, r_b: str) -> str:
    """Toy judge: prefer non-empty, then complete reasoning, then length cap."""
    a = r_a.strip()
    b = r_b.strip()
    # Penalize obvious wrong patterns
    def score(t: str) -> float:
        s = 0.0
        if "answer" in t.lower() or "</think>" in t:
            s += 0.4
        if "step" in t.lower() or ":" in t:
            s += 0.2
        # length: cap at 200 chars
        s += min(0.3, len(t) / 600)
        # leak penalty
        if "step 1: get materials" in t.lower():
            s -= 0.8
        return s
    sa, sb = score(a), score(b)
    if abs(sa - sb) < 0.05:
        return "tie"
    return "A" if sa > sb else "B"


def run_round_robin() -> List[Battle]:
    zoo = load_all()
    keys = list(zoo.keys())
    battles: List[Battle] = []
    for a, b in itertools.combinations(keys, 2):
        for q in KEY_QUESTIONS:
            r_a = zoo[a].generate(q)
            r_b = zoo[b].generate(q)
            winner = _length_quality_judge(r_a, r_b)
            battles.append(Battle(q, a, b, winner))
            # Swap order too
            winner_swap = _length_quality_judge(r_b, r_a)
            battles.append(Battle(q, b, a, winner_swap))
    return battles


def fit_bt(battles: List[Battle], n_iter: int = 200) -> Dict[str, float]:
    """MM algorithm for BT."""
    models = sorted({b.a for b in battles} | {b.b for b in battles})
    log_s = {m: 0.0 for m in models}
    W: Dict[Tuple[str, str], float] = {}
    for b in battles:
        if b.winner == "A":
            W[(b.a, b.b)] = W.get((b.a, b.b), 0) + 1
        elif b.winner == "B":
            W[(b.b, b.a)] = W.get((b.b, b.a), 0) + 1
        else:
            W[(b.a, b.b)] = W.get((b.a, b.b), 0) + 0.5
            W[(b.b, b.a)] = W.get((b.b, b.a), 0) + 0.5
    for _ in range(n_iter):
        s = {m: math.exp(log_s[m]) for m in models}
        new_s = {}
        for i in models:
            W_i = sum(W.get((i, j), 0) for j in models if j != i)
            denom = 0.0
            for j in models:
                if i == j:
                    continue
                n_ij = W.get((i, j), 0) + W.get((j, i), 0)
                if n_ij > 0:
                    denom += n_ij / (s[i] + s[j])
            new_s[i] = (W_i / denom) if denom > 0 else s[i]
        log_s = {m: math.log(max(1e-9, new_s[m])) for m in models}
        mean = sum(log_s.values()) / len(log_s)
        log_s = {m: v - mean for m, v in log_s.items()}
    return log_s


def to_elo(log_s: Dict[str, float], base: int = 1500) -> Dict[str, int]:
    scale = 400 / math.log(10)
    return {m: int(base + scale * v) for m, v in log_s.items()}


def run_capstone_arena() -> Dict:
    battles = run_round_robin()
    bt = fit_bt(battles)
    elo = to_elo(bt)
    ranked = sorted(elo.items(), key=lambda kv: -kv[1])
    return {"n_battles": len(battles), "elo": elo, "ranking": ranked}


def to_md(report: Dict) -> str:
    lines = ["# mini-Arena (5 ckpt round-robin)", "",
             f"Total battles: {report['n_battles']}", "",
             "| rank | ckpt | Elo |", "|---|---|---:|"]
    for i, (name, e) in enumerate(report["ranking"], 1):
        lines.append(f"| {i} | {name} | {e} |")
    return "\n".join(lines)


def _self_test() -> int:
    report = run_capstone_arena()
    elo = report["elo"]
    assert set(elo.keys()) == {"vanilla", "lora", "dpo", "r1_tiny", "phi_tiny"}
    # vanilla likely lowest (gives wrong answer 23 + leaks)
    assert elo["vanilla"] == min(elo.values()), elo
    md = to_md(report)
    assert "mini-Arena" in md
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"mini_arena.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
    print(to_md(run_capstone_arena()))
