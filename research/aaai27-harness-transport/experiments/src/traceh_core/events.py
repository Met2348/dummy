"""Deterministic event detectors over recorded baseline traces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _prefix_score(entry: Mapping[str, Any]) -> float:
    predicates = entry["prefix_state"].get("completed_predicates", [])
    for predicate in predicates:
        text = str(predicate)
        if text.startswith("score="):
            return float(text.split("=", 1)[1])
    raise ValueError("trace prefix is missing score predicate")


def find_no_progress_prefixes(
    trace: Sequence[Mapping[str, Any]],
    *,
    revisit_threshold: int = 3,
    window: int = 12,
) -> list[dict[str, Any]]:
    """Emit the first same-observation/same-score revisit crossing per state key."""

    if revisit_threshold < 2:
        raise ValueError("revisit_threshold must be at least two")
    if window < revisit_threshold:
        raise ValueError("window must be at least revisit_threshold")
    emitted: set[tuple[str, float]] = set()
    events: list[dict[str, Any]] = []
    for index, entry in enumerate(trace):
        observation = str(entry["prefix_state"]["recent_tool_result"]["observation"])
        score = _prefix_score(entry)
        key = (observation, score)
        recent = trace[max(0, index - window + 1) : index + 1]
        revisit_count = sum(
            str(item["prefix_state"]["recent_tool_result"]["observation"]) == observation
            and _prefix_score(item) == score
            for item in recent
        )
        if revisit_count >= revisit_threshold and key not in emitted:
            events.append(
                {
                    "step_index": int(entry["step_index"]),
                    "prefix_state_hash": str(entry["prefix_state_hash"]),
                    "event_type": "no_progress",
                    "revisit_count": revisit_count,
                    "window": window,
                }
            )
            emitted.add(key)
    return events

