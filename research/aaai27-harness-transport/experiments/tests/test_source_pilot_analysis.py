from __future__ import annotations

from traceh_core.source_pilot_analysis import (
    combine_source_pilots,
    compare_action_traces,
    summarize_model_source_pilot,
)


def test_compare_action_traces_reports_first_divergence_and_count() -> None:
    none_trace = [
        {"selected_action": "look"},
        {"selected_action": "go to desk"},
        {"selected_action": "open desk"},
    ]
    replan_trace = [
        {"selected_action": "look"},
        {"selected_action": "inventory"},
        {"selected_action": "open desk"},
    ]
    result = compare_action_traces(none_trace, replan_trace)
    assert result == {
        "changed": True,
        "first_divergence": 1,
        "different_action_count": 1,
        "none_steps": 3,
        "replan_steps": 3,
    }


def test_compare_action_traces_counts_unpaired_tail_steps() -> None:
    result = compare_action_traces(
        [{"selected_action": "look"}],
        [{"selected_action": "look"}, {"selected_action": "inventory"}],
    )
    assert result["first_divergence"] == 1
    assert result["different_action_count"] == 1


def test_combine_source_pilots_pivots_when_both_models_have_zero_utility() -> None:
    summaries = [
        {
            "model_id": "Qwen/Qwen3-4B",
            "integrity_ok": True,
            "pair_count": 10,
            "all_raw_scores_zero": True,
        },
        {
            "model_id": "Qwen/Qwen3-8B",
            "integrity_ok": True,
            "pair_count": 7,
            "all_raw_scores_zero": True,
        },
    ]
    result = combine_source_pilots(summaries)
    assert result["decision"] == "PIVOT_SOURCE_POLICY_OR_TASK_SIGNAL"
    assert result["total_pair_count"] == 17
    assert result["transport_tested"] is False


def test_summarize_model_source_pilot_separates_behavior_change_from_utility() -> None:
    report = {
        "frozen_config": {"model_id": "Qwen/Qwen3-4B"},
        "integrity": {"ok": True},
        "pairs": [
            {
                "candidate_id": "c1",
                "seed": 0,
                "none_score": 0.0,
                "replan_score": 0.0,
            }
        ],
    }
    paired_traces = [
        {
            "candidate_id": "c1",
            "seed": 0,
            "none": [{"selected_action": "look"}],
            "replan": [{"selected_action": "inventory"}],
            "plan": "Inspect inventory first.",
            "none_extra_model_calls": 0,
            "replan_extra_model_calls": 1,
            "parser_failure": False,
        }
    ]
    result = summarize_model_source_pilot(report, paired_traces)
    assert result["changed_pair_count"] == 1
    assert result["positive_utility_pair_count"] == 0
    assert result["all_raw_scores_zero"] is True
