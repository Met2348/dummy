from __future__ import annotations

from traceh_core.branch_analysis import decide_branch_pilot


def test_decision_moves_to_8b_when_all_raw_scores_are_zero() -> None:
    pairs = [
        {"candidate_id": "c1", "seed": seed, "none_score": 0.0, "replan_score": 0.0}
        for seed in (0, 1, 2)
    ]
    result = decide_branch_pilot(pairs, integrity_ok=True)
    assert result["decision"] == "MOVE_TO_QWEN3_8B"
    assert result["positive_seed0_candidates"] == 0


def test_decision_uses_explicit_escalation_after_8b_all_zero() -> None:
    pairs = [
        {"candidate_id": "c1", "seed": seed, "none_score": 0.0, "replan_score": 0.0}
        for seed in (0, 1, 2)
    ]
    result = decide_branch_pilot(
        pairs,
        integrity_ok=True,
        all_zero_decision="PIVOT_SOURCE_POLICY_OR_TASK_SIGNAL",
    )
    assert result["decision"] == "PIVOT_SOURCE_POLICY_OR_TASK_SIGNAL"


def test_decision_goes_when_two_seed0_candidates_improve_and_repeats_are_stable() -> None:
    pairs = [
        {"candidate_id": candidate, "seed": seed, "none_score": 0.0, "replan_score": 1.0}
        for candidate in ("c1", "c2")
        for seed in (0, 1, 2)
    ]
    result = decide_branch_pilot(pairs, integrity_ok=True)
    assert result["decision"] == "GO_EXPAND_FOUR_ACTION"
    assert result["positive_seed0_candidates"] == 2
    assert result["repeat_sign_stability"] == 1.0


def test_decision_blocks_when_none_integrity_fails() -> None:
    result = decide_branch_pilot([], integrity_ok=False)
    assert result["decision"] == "BLOCKED_NONE_INTEGRITY"
