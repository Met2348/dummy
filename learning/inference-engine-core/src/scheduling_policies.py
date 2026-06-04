"""FCFS / SJF / priority pickers — pluggable into Engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Deque, List
from common import Request


def fcfs_picker(pending: Deque[Request]) -> Request:
    return pending.popleft()


def sjf_picker(pending: Deque[Request]) -> Request:
    """Shortest-job-first by total expected length."""
    best_i = min(range(len(pending)), key=lambda i: len(pending[i].prompt_ids) + pending[i].max_new_tokens)
    best = pending[best_i]
    del pending[best_i]
    return best


def priority_picker(pending: Deque[Request]) -> Request:
    """Picks highest priority (stored as attribute `priority`, default 0)."""
    best_i = max(range(len(pending)), key=lambda i: getattr(pending[i], "priority", 0))
    best = pending[best_i]
    del pending[best_i]
    return best


PICKERS = {
    "fcfs": fcfs_picker,
    "sjf": sjf_picker,
    "priority": priority_picker,
}


def schedule_order(reqs: List[Request], policy: str) -> List[int]:
    """Return the request id order that `policy` would dispatch."""
    from collections import deque
    pending = deque(reqs)
    picker = PICKERS[policy]
    return [picker(pending).rid for _ in range(len(reqs))]
