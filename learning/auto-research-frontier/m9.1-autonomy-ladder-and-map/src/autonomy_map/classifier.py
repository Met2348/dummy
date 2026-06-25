"""分级器：只用证据推 evidenced_level，再和自称对比出 hype gap。

阶梯定义（综述 2505.13259 的 Tool→Analyst→Scientist）：
- **Tool**：只自动化单环/少数非核心环（如只检索、只写作）。
- **Analyst**：自动化"实验+分析"核心，或一条≥3 环的链，但**问题由人给定**。
- **Scientist**：**自己定问题**且闭合"创意→实验→分析"这条核心链。
"""
from __future__ import annotations

from dataclasses import dataclass

from .systems import STAGES, System

LEVELS = ("tool", "analyst", "scientist")
LEVEL_RANK = {lvl: i for i, lvl in enumerate(LEVELS)}

_IDEATION = {"ideation", "hypothesis"}   # "提出问题/假设"这一类环


@dataclass(frozen=True)
class Classification:
    name: str
    claimed_level: str
    evidenced_level: str
    coverage: tuple            # 自动化的阶段（按 STAGES 顺序）
    hype_gap: int              # claimed_rank - evidenced_rank（>0 = 自称夸大）
    self_verified_only: bool   # 结果只靠自评（无独立验证）→ 可信度警示
    reasons: tuple


def evidenced_level(sys: System) -> str:
    """★ 决策线（留给你打磨 / hands-on）：到底几环、闭不闭环、问题谁定，才算升一级？

    当前规则刻意简单透明，方便你改了之后亲眼看到全景地图怎么变。
    """
    cov = set(sys.automates)
    has_ideation = bool(cov & _IDEATION)
    has_core_exec = "experiment" in cov and "analysis" in cov
    closes_loop = has_ideation and has_core_exec

    if closes_loop and not sys.human_sets_problem:
        return "scientist"
    if has_core_exec or len(cov) >= 3:
        return "analyst"
    return "tool"


def classify(sys: System) -> Classification:
    ev = evidenced_level(sys)
    gap = LEVEL_RANK[sys.claimed_level] - LEVEL_RANK[ev]

    cov_ordered = tuple(s for s in STAGES if s in set(sys.automates))
    reasons = [
        f"自动化 {len(cov_ordered)} 环（{','.join(cov_ordered)}）",
        "问题由人给定" if sys.human_sets_problem else "自己定问题/假设",
        "有独立验证" if sys.independent_verification else "仅自评（无独立验证）",
    ]
    if gap > 0:
        reasons.append(f"[!] 自称 {sys.claimed_level} > 证据级别 {ev}（hype gap +{gap}）")

    return Classification(
        name=sys.name,
        claimed_level=sys.claimed_level,
        evidenced_level=ev,
        coverage=cov_ordered,
        hype_gap=gap,
        self_verified_only=not sys.independent_verification,
        reasons=tuple(reasons),
    )
