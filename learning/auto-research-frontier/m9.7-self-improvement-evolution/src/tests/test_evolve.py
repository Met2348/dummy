"""V2 测试：锁死 9.7 的诚实性——进化确定性、优化代理=刷分(holdout 不动)、
优化真目标=真涨、代理-真目标缺口的反差。
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from selfimprove import (
    SEED, Genome, evolve, holdout_fitness, naive_fitness,
)


def test_evolution_deterministic():
    a = evolve(naive_fitness)[0]
    b = evolve(naive_fitness)[0]
    assert a == b


def test_memorize_inflates_naive_not_holdout():
    """背一个泄漏测试点：naive 涨、holdout 不变（memo 不碰 held-out）。"""
    from selfimprove.genome import LEAKED, true_label
    x, y = next((px, py) for px, py in LEAKED
                if SEED.predict(px, py) != true_label(px, py))
    cheated = Genome(SEED.threshold, frozenset({(x, y, true_label(x, y))}))
    assert naive_fitness(cheated) > naive_fitness(SEED)
    assert holdout_fitness(cheated) == holdout_fitness(SEED)


def test_optimizing_proxy_is_reward_hacking():
    """优化 naive：naive 真涨，holdout 一点没动——典型 reward hacking。"""
    best, _ = evolve(naive_fitness)
    assert naive_fitness(best) > naive_fitness(SEED)
    assert holdout_fitness(best) == holdout_fitness(SEED)   # 真目标纹丝不动
    assert best.threshold == SEED.threshold                 # 没真改进，只是背书
    assert len(best.memo) > 0


def test_optimizing_true_goal_really_improves():
    """优化 holdout：真目标真涨，且靠的是调 threshold（真泛化），不是背书。"""
    best, _ = evolve(holdout_fitness)
    assert holdout_fitness(best) > holdout_fitness(SEED) + 0.2
    assert best.threshold != SEED.threshold
    assert len(best.memo) == 0


def test_proxy_true_gap_contrast():
    """优化代理后 naive≫holdout（缺口大）；优化真目标后两者贴合（缺口小）。"""
    bn, _ = evolve(naive_fitness)
    bh, _ = evolve(holdout_fitness)
    gap_proxy = naive_fitness(bn) - holdout_fitness(bn)
    gap_true = naive_fitness(bh) - holdout_fitness(bh)
    assert gap_proxy > 0.3
    assert gap_true <= gap_proxy - 0.3


def test_archive_grows_monotonically_in_its_own_fitness():
    """档案里每次纳入的新解，其被优化的 fitness 必须严格更高（keep-if-better）。"""
    _, arch = evolve(naive_fitness)
    fits = [rec[2] for rec in arch]      # rec=(gen, g, fitness值, naive, holdout)
    assert all(b > a for a, b in zip(fits, fits[1:]))


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
