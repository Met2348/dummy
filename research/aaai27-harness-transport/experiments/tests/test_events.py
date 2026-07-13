from __future__ import annotations

from traceh_core.events import find_no_progress_prefixes


def entry(step: int, observation: str, score: float = 0.0) -> dict:
    return {
        "step_index": step,
        "prefix_state_hash": f"hash-{step}",
        "prefix_state": {
            "completed_predicates": [f"won=False", f"score={score:.6f}"],
            "recent_tool_result": {"observation": observation},
        },
    }


def test_no_progress_detects_third_same_state_and_score_in_window() -> None:
    trace = [entry(0, "A"), entry(1, "B"), entry(2, "A"), entry(3, "B"), entry(4, "A")]
    events = find_no_progress_prefixes(trace, revisit_threshold=3, window=5)
    assert events == [
        {
            "step_index": 4,
            "prefix_state_hash": "hash-4",
            "event_type": "no_progress",
            "revisit_count": 3,
            "window": 5,
        }
    ]


def test_no_progress_does_not_merge_same_observation_after_score_gain() -> None:
    trace = [entry(0, "A", 0), entry(1, "A", 0), entry(2, "A", 1)]
    assert find_no_progress_prefixes(trace, revisit_threshold=3, window=5) == []

