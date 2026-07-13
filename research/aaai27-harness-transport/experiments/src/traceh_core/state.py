"""Canonical state hashing and field-level replay diffs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


STATE_HASH_FIELDS = (
    "task_id",
    "location",
    "inventory",
    "completed_predicates",
    "admissible_commands",
    "step_index",
    "recent_tool_result",
)


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (set, frozenset)):
        return sorted((_normalize(item) for item in value), key=_sort_key)
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("state hash rejects NaN and infinity")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported state value: {type(value).__name__}")


def _sort_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_state(state: Mapping[str, Any], schema_version: str = "traceh-state-v1") -> dict[str, Any]:
    payload = {field: state.get(field) for field in STATE_HASH_FIELDS}
    payload["schema_version"] = schema_version
    for field in ("inventory", "completed_predicates", "admissible_commands"):
        value = payload.get(field)
        if isinstance(value, (list, tuple, set, frozenset)):
            payload[field] = sorted((_normalize(item) for item in value), key=_sort_key)
    return _normalize(payload)


def state_hash(state: Mapping[str, Any], schema_version: str = "traceh-state-v1") -> str:
    encoded = json.dumps(
        canonical_state(state, schema_version),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def diff_states(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    a = canonical_state(left)
    b = canonical_state(right)
    return {
        key: {"left": a.get(key), "right": b.get(key)}
        for key in sorted(set(a) | set(b))
        if a.get(key) != b.get(key)
    }

