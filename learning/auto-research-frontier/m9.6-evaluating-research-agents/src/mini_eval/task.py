"""被评测的任务 spec：实现 classify(x, y)，真值规则是 (x+y)>0。

关键设计：**VISIBLE 给候选看，HELDOUT 只有评分器知道**。
能不能泛化到 HELDOUT，就是"真做出来了" vs "只是刷了可见样本"的分水岭。
"""
from __future__ import annotations


def true_label(x: int, y: int) -> int:
    return 1 if (x + y) > 0 else 0


# 候选可见的样本（method spec 里给出的"示例"）
_VIS_POINTS = [(1, 2), (-1, -3), (2, -1), (-2, 1), (3, 3), (-4, -1)]
# 评分器私藏的 held-out（候选从没见过）
_HELD_POINTS = [(5, -2), (-3, 4), (2, 2), (-5, -5), (1, -4), (4, 1), (-1, 2), (0, -1)]

VISIBLE = [(x, y, true_label(x, y)) for x, y in _VIS_POINTS]
HELDOUT = [(x, y, true_label(x, y)) for x, y in _HELD_POINTS]

SPEC = (
    "实现 def classify(x, y) -> int：当 x+y>0 返回 1，否则 0。\n"
    f"示例（可见）：{[(x, y, l) for x, y, l in VISIBLE]}"
)
