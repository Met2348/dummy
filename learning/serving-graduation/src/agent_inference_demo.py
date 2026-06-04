"""Agent inference — naive vs radix-cached, multi-turn.

Naively each turn re-prefills the entire history.  With radix caching only
the new user input + tool output needs prefilling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class TurnStats:
    naive_prefill_tokens: int = 0
    cached_prefill_tokens: int = 0


def simulate_agent_turn(history_len: int, new_user_tokens: int, cached: bool) -> int:
    if cached:
        return new_user_tokens
    return history_len + new_user_tokens


def run_multi_turn(turns: int = 5, base_history: int = 200, per_turn_new: int = 50) -> TurnStats:
    naive = 0
    cached = 0
    history = base_history
    for t in range(turns):
        naive += simulate_agent_turn(history, per_turn_new, cached=False)
        cached += simulate_agent_turn(history, per_turn_new, cached=True)
        history += per_turn_new + 30   # tool output + assistant reply
    return TurnStats(naive_prefill_tokens=naive, cached_prefill_tokens=cached)
