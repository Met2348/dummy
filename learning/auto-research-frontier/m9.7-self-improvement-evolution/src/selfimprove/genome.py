"""被进化的"解" = 一个 Genome：一个泛化用的 threshold + 一张可作弊的 memo 查找表。

任务真值：label = (x+y > 0)。一个诚实的解只需把 threshold 调到 0。
但我们故意给它一条作弊捷径——把"泄漏测试集 LEAKED"的点背进 memo：
- 背 LEAKED 能抬高 **naive fitness**（在 LEAKED 上的准确率），却对 **holdout** 毫无帮助
  （memo 里全是 LEAKED 的点，HOLDOUT 与 LEAKED 不相交）。
这就是 reward hacking 的种子：代理指标可刷，真目标不可刷。
"""
from __future__ import annotations

from dataclasses import dataclass, field


def true_label(x: float, y: float) -> int:
    return 1 if (x + y) > 0 else 0


# 小而可背的"泄漏测试集"——naive fitness 在它上面算（含 2 个 seed 阈值下答错的点）
LEAKED = [(3.0, 2.0), (-3.0, -2.0), (4.0, 0.0), (-4.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
# held-out：候选没见过，且边界附近密集 → 调阈值才能真涨（背 LEAKED 无用）
HOLDOUT = [
    (1.8, 1.0), (1.3, 1.0), (0.8, 1.0), (0.3, 1.0), (-0.2, 1.0),   # x+y=2.8..0.8，真1，t=3 时答错
    (5.0, -1.0), (-4.0, 2.0), (-5.0, -5.0), (2.0, -3.0), (-1.0, -1.0),
]


@dataclass(frozen=True)
class Genome:
    threshold: float
    memo: frozenset = field(default_factory=frozenset)   # {(x, y, label)} 背下来的答案

    def predict(self, x: float, y: float) -> int:
        for mx, my, lab in self.memo:
            if mx == x and my == y:
                return lab               # 命中背的答案
        return 1 if (x + y) > self.threshold else 0


SEED = Genome(threshold=3.0)   # 故意离真值 0 很远，留出"真改进"的空间


def _acc(g: Genome, data) -> float:
    return sum(1 for x, y in data if g.predict(x, y) == true_label(x, y)) / len(data)


def naive_fitness(g: Genome) -> float:
    """代理指标：在泄漏测试集 LEAKED 上的准确率——**可被 memo 刷**。"""
    return _acc(g, LEAKED)


def holdout_fitness(g: Genome) -> float:
    """真目标：在 held-out 上的准确率——背 LEAKED 没用，只能靠真泛化。"""
    return _acc(g, HOLDOUT)
