"""进化档案循环：变异自己 → 按 fitness 评估 → keep-if-better → 存档案。

这是 DGM/ADAS/AlphaEvolve 那一支"自我改进"的最小骨架。
变异是**确定性枚举**（不用随机数），所以整条进化轨迹可复现、可测试。
"""
from __future__ import annotations

from .genome import LEAKED, SEED, Genome, holdout_fitness, naive_fitness, true_label


def _mutations(g: Genome) -> list:
    """枚举候选变异：阈值 ±0.5（真改进的杠杆）+ 背下一个还没背的 LEAKED 点（作弊的杠杆）。"""
    cands = [Genome(round(g.threshold + 0.5, 3), g.memo),
             Genome(round(g.threshold - 0.5, 3), g.memo)]
    memoed = {(x, y) for x, y, _ in g.memo}
    # 作弊杠杆：背下第一个"当前答错且还没背"的 LEAKED 点（聪明的自改进者会去背自己的失败）
    for x, y in LEAKED:
        if (x, y) not in memoed and g.predict(x, y) != true_label(x, y):
            cands.append(Genome(g.threshold, g.memo | {(x, y, true_label(x, y))}))
            break          # 一次只背一个，让"作弊"和"真改进"公平竞争每一代
    return cands


def evolve(fitness, gens: int = 14, seed: Genome = None):
    """对给定 fitness 做贪心进化。返回 (最终 best, 档案)。

    档案每条记 (gen, genome, 该fitness值, naive值, holdout值)，便于看"代理涨/真目标动没动"。
    """
    best = seed if seed is not None else SEED
    record = lambda gen, g: (gen, g, round(fitness(g), 3),
                             round(naive_fitness(g), 3), round(holdout_fitness(g), 3))
    archive = [record(0, best)]
    for gen in range(1, gens + 1):
        cands = _mutations(best)
        # 贪心选 fitness 最高的候选；平手取枚举靠前者（阈值变异在前 → 不偏向作弊）
        scored = sorted(((fitness(c), i, c) for i, c in enumerate(cands)),
                        key=lambda t: (-t[0], t[1]))
        bf, _, bc = scored[0]
        if bf > fitness(best) + 1e-12:
            best = bc
            archive.append(record(gen, best))
    return best, archive
