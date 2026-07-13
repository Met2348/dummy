"""TRACE-H local engineering primitives."""

from .agent import (
    ActionParseResult,
    ACTION_LABELS,
    allowed_next_tokens,
    build_action_messages,
    build_indexed_action_messages,
    build_labeled_action_messages,
    parse_admissible_action,
    parse_admissible_action_id,
    parse_normalized_action_id,
    parse_action_label,
)
from .baseline import GenerationResult, run_baseline_episode
from .branch import build_replan_messages, run_episode_continuation
from .branch_analysis import decide_branch_pilot
from .contracts import (
    ACTION_CONTRACTS,
    ActionDecision,
    EventType,
    HarnessAction,
    RequestContext,
    resolve_action_request,
)
from .events import find_no_progress_prefixes
from .records import AppendOnlyRecordStore, DuplicateRunIdError
from .state import diff_states, state_hash

__all__ = [
    "ACTION_CONTRACTS",
    "ACTION_LABELS",
    "allowed_next_tokens",
    "ActionParseResult",
    "ActionDecision",
    "AppendOnlyRecordStore",
    "DuplicateRunIdError",
    "EventType",
    "GenerationResult",
    "HarnessAction",
    "RequestContext",
    "build_action_messages",
    "build_indexed_action_messages",
    "build_labeled_action_messages",
    "build_replan_messages",
    "diff_states",
    "decide_branch_pilot",
    "find_no_progress_prefixes",
    "resolve_action_request",
    "parse_admissible_action",
    "parse_admissible_action_id",
    "parse_normalized_action_id",
    "parse_action_label",
    "run_baseline_episode",
    "run_episode_continuation",
    "state_hash",
]
