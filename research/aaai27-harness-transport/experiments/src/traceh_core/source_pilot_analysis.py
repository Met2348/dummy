"""Cross-model analysis for local source-policy branch pilots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import zip_longest
from typing import Any


def compare_action_traces(
    none_trace: Sequence[Mapping[str, Any]],
    replan_trace: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare aligned selected actions, including any unmatched tail."""

    sentinel = object()
    none_actions = [item["selected_action"] for item in none_trace]
    replan_actions = [item["selected_action"] for item in replan_trace]
    differing = [
        index
        for index, (none_action, replan_action) in enumerate(
            zip_longest(none_actions, replan_actions, fillvalue=sentinel)
        )
        if none_action != replan_action
    ]
    return {
        "changed": bool(differing),
        "first_divergence": differing[0] if differing else None,
        "different_action_count": len(differing),
        "none_steps": len(none_actions),
        "replan_steps": len(replan_actions),
    }


def summarize_model_source_pilot(
    report: Mapping[str, Any],
    paired_traces: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize whether REPLAN changed behavior and terminal utility."""

    expected_keys = {
        (str(pair["candidate_id"]), int(pair["seed"])) for pair in report["pairs"]
    }
    observed_keys = {
        (str(pair["candidate_id"]), int(pair["seed"])) for pair in paired_traces
    }
    if expected_keys != observed_keys:
        raise ValueError("paired traces do not match report candidate/seed keys")

    report_pairs = {
        (str(pair["candidate_id"]), int(pair["seed"])): pair
        for pair in report["pairs"]
    }
    rows = []
    for paired in paired_traces:
        key = (str(paired["candidate_id"]), int(paired["seed"]))
        scores = report_pairs[key]
        trace_effect = compare_action_traces(paired["none"], paired["replan"])
        utility_effect = float(scores["replan_score"]) - float(scores["none_score"])
        rows.append(
            {
                "candidate_id": key[0],
                "seed": key[1],
                **trace_effect,
                "none_score": float(scores["none_score"]),
                "replan_score": float(scores["replan_score"]),
                "raw_score_advantage": utility_effect,
                "plan_nonempty": bool(str(paired.get("plan") or "").strip()),
                "plan_character_count": len(str(paired.get("plan") or "")),
                "none_extra_model_calls": int(paired["none_extra_model_calls"]),
                "replan_extra_model_calls": int(paired["replan_extra_model_calls"]),
                "parser_failure": bool(paired["parser_failure"]),
            }
        )

    return {
        "model_id": str(report["frozen_config"]["model_id"]),
        "integrity_ok": bool(report["integrity"]["ok"]),
        "pair_count": len(rows),
        "changed_pair_count": sum(row["changed"] for row in rows),
        "unchanged_pair_count": sum(not row["changed"] for row in rows),
        "positive_utility_pair_count": sum(
            row["raw_score_advantage"] > 1e-12 for row in rows
        ),
        "negative_utility_pair_count": sum(
            row["raw_score_advantage"] < -1e-12 for row in rows
        ),
        "all_raw_scores_zero": all(
            abs(row["none_score"]) <= 1e-12
            and abs(row["replan_score"]) <= 1e-12
            for row in rows
        ),
        "all_plans_nonempty": bool(rows) and all(row["plan_nonempty"] for row in rows),
        "protocol_ok": bool(rows)
        and all(
            row["none_extra_model_calls"] == 0
            and row["replan_extra_model_calls"] == 1
            and not row["parser_failure"]
            for row in rows
        ),
        "pairs": rows,
    }


def combine_source_pilots(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the cross-model source-policy stop/pivot rule."""

    if not summaries:
        decision = "BLOCKED_NO_SOURCE_RESULTS"
    elif not all(bool(summary["integrity_ok"]) for summary in summaries):
        decision = "BLOCKED_SOURCE_INTEGRITY"
    elif all(bool(summary["all_raw_scores_zero"]) for summary in summaries):
        decision = "PIVOT_SOURCE_POLICY_OR_TASK_SIGNAL"
    elif any(int(summary.get("positive_utility_pair_count", 0)) > 0 for summary in summaries):
        decision = "REVIEW_NONZERO_SOURCE_EFFECTS"
    else:
        decision = "INCONCLUSIVE_SOURCE_PILOT"
    return {
        "decision": decision,
        "source_model_count": len(summaries),
        "total_pair_count": sum(int(summary["pair_count"]) for summary in summaries),
        "all_integrity_ok": bool(summaries)
        and all(bool(summary["integrity_ok"]) for summary in summaries),
        "all_raw_scores_zero": bool(summaries)
        and all(bool(summary["all_raw_scores_zero"]) for summary in summaries),
        "transport_tested": False,
    }
