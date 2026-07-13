"""Predeclared decision rules for the local NONE/REPLAN branch pilot."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def _sign(value: float, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def decide_branch_pilot(
    pairs: Sequence[Mapping[str, Any]],
    *,
    integrity_ok: bool,
    all_zero_decision: str = "MOVE_TO_QWEN3_8B",
) -> dict[str, Any]:
    """Apply the frozen local Go/escalate/Pivot decision rule."""

    if not integrity_ok:
        return {
            "decision": "BLOCKED_NONE_INTEGRITY",
            "positive_seed0_candidates": 0,
            "negative_seed0_candidates": 0,
            "repeat_sign_stability": 0.0,
        }

    effects = [float(item["replan_score"]) - float(item["none_score"]) for item in pairs]
    seed0 = [
        float(item["replan_score"]) - float(item["none_score"])
        for item in pairs
        if int(item["seed"]) == 0
    ]
    positive = sum(_sign(effect) > 0 for effect in seed0)
    negative = sum(_sign(effect) < 0 for effect in seed0)

    grouped: dict[str, list[int]] = defaultdict(list)
    for item, effect in zip(pairs, effects, strict=True):
        grouped[str(item["candidate_id"])].append(_sign(effect))
    repeated_groups = [signs for signs in grouped.values() if len(signs) > 1]
    if repeated_groups:
        stability = sum(
            max(Counter(signs).values()) / len(signs) for signs in repeated_groups
        ) / len(repeated_groups)
    else:
        stability = 0.0

    all_scores_zero = all(
        abs(float(item["none_score"])) <= 1e-12
        and abs(float(item["replan_score"])) <= 1e-12
        for item in pairs
    )
    if all_scores_zero:
        decision = all_zero_decision
    elif positive >= 2 and stability >= 0.70:
        decision = "GO_EXPAND_FOUR_ACTION"
    elif positive == 0 and negative > 0:
        decision = "PIVOT_REPLAN"
    else:
        decision = "INCONCLUSIVE"
    return {
        "decision": decision,
        "positive_seed0_candidates": positive,
        "negative_seed0_candidates": negative,
        "repeat_sign_stability": stability,
        "all_raw_scores_zero": all_scores_zero,
    }
