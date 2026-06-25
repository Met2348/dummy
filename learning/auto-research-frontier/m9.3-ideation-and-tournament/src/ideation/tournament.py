"""锦标赛：用成对评判跑一个确定性 Elo，把点子排序（复用 llm-judge-arena 思路）。

外加 rank_by_feasibility——用**真实执行结果**排另一份榜，两份榜一对比，
ideation-execution gap 就现形了。
"""
from __future__ import annotations

from .judge import judge_score, prefer
from .task import train_logreg


def run_tournament(ideas, use_self_bias: bool = True, passes: int = 8,
                   k: float = 24.0, base: float = 1000.0):
    """全员循环赛、多轮，确定性 Elo。返回 (elo dict, 排名 id 列表)。"""
    elo = {it.id: base for it in ideas}
    idx = {it.id: it for it in ideas}
    ids = [it.id for it in ideas]
    for _ in range(passes):
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = idx[ids[i]], idx[ids[j]]
                ea = 1.0 / (1.0 + 10 ** ((elo[b.id] - elo[a.id]) / 400.0))
                winner = prefer(a, b, use_self_bias)
                sa = 1.0 if winner == a.id else 0.0
                elo[a.id] += k * (sa - ea)
                elo[b.id] += k * ((1.0 - sa) - (1.0 - ea))
    ranking = sorted(ids, key=lambda x: (-elo[x], x))
    return elo, ranking


def feasibility_of(ideas) -> dict:
    """每个点子真跑 toy 任务，得真实可行性（准确率）。确定性。"""
    return {it.id: train_logreg(it.config) for it in ideas}


def rank_by_feasibility(ideas) -> list:
    feas = feasibility_of(ideas)
    return sorted((it.id for it in ideas), key=lambda x: (-feas[x], x))
