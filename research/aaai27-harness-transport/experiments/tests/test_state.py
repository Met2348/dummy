from __future__ import annotations

import pytest

from traceh_core.state import diff_states, state_hash


def _state() -> dict:
    return {
        "task_id": "task-001",
        "location": "kitchen",
        "inventory": ["apple", "knife"],
        "completed_predicates": {"opened_fridge", "found_apple"},
        "admissible_commands": ["take apple", "go north"],
        "step_index": 4,
        "recent_tool_result": {"ok": True, "message": "fridge open"},
        "ephemeral_wall_time": 9.9,
    }


def test_hash_ignores_container_order_and_ephemeral_fields() -> None:
    left = _state()
    right = _state()
    right["inventory"] = ["knife", "apple"]
    right["admissible_commands"] = ["go north", "take apple"]
    right["ephemeral_wall_time"] = 1000
    assert state_hash(left) == state_hash(right)


def test_hash_changes_on_execution_state_change() -> None:
    left = _state()
    right = _state()
    right["step_index"] = 5
    assert state_hash(left) != state_hash(right)
    assert diff_states(left, right) == {"step_index": {"left": 4, "right": 5}}


def test_hash_rejects_non_finite_values() -> None:
    state = _state()
    state["recent_tool_result"] = {"score": float("nan")}
    with pytest.raises(ValueError):
        state_hash(state)

