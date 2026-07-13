"""Executable TRACE-H harness-action contracts.

These contracts intentionally encode the narrow development semantics from the
local experiment plan. They are policy-independent: an action request is either
legal in the current event/context or deterministically falls back to NONE.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class HarnessAction(str, Enum):
    NONE = "NONE"
    CHECK = "CHECK"
    RETRY = "RETRY"
    REPLAN = "REPLAN"


class EventType(str, Enum):
    INVALID_REJECTED = "invalid_rejected"
    NO_PROGRESS = "no_progress"
    MISSING_PRECONDITION = "missing_precondition"
    PRE_SUBMIT_UNCERTAINTY = "pre_submit_uncertainty"
    BUDGET_PRESSURE = "budget_pressure"


@dataclass(frozen=True)
class ActionContract:
    action_id: HarnessAction
    version: str
    description: str
    allowed_events: tuple[EventType, ...]
    max_extra_model_calls: int
    max_extra_env_steps: int
    max_added_tokens: int
    read_only: bool
    adds_oracle_information: bool = False

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["action_id"] = self.action_id.value
        raw["allowed_events"] = [event.value for event in self.allowed_events]
        return raw


ALL_EVENTS = tuple(EventType)

ACTION_CONTRACTS: dict[HarnessAction, ActionContract] = {
    HarnessAction.NONE: ActionContract(
        action_id=HarnessAction.NONE,
        version="0.1.0",
        description="Continue with the reference harness without extra control.",
        allowed_events=ALL_EVENTS,
        max_extra_model_calls=0,
        max_extra_env_steps=0,
        max_added_tokens=0,
        read_only=True,
    ),
    HarnessAction.CHECK: ActionContract(
        action_id=HarnessAction.CHECK,
        version="0.1.0",
        description="Inject a read-only structured check of visible execution state.",
        allowed_events=(
            EventType.NO_PROGRESS,
            EventType.MISSING_PRECONDITION,
            EventType.PRE_SUBMIT_UNCERTAINTY,
        ),
        max_extra_model_calls=0,
        max_extra_env_steps=0,
        max_added_tokens=128,
        read_only=True,
    ),
    HarnessAction.RETRY: ActionContract(
        action_id=HarnessAction.RETRY,
        version="0.1.0",
        description="Normalize and replay one rejected or repeated environment action.",
        allowed_events=(EventType.INVALID_REJECTED, EventType.NO_PROGRESS),
        max_extra_model_calls=0,
        max_extra_env_steps=1,
        max_added_tokens=0,
        read_only=False,
    ),
    HarnessAction.REPLAN: ActionContract(
        action_id=HarnessAction.REPLAN,
        version="0.1.0",
        description="Ask the same executor for a plan of at most three remaining steps.",
        allowed_events=(
            EventType.NO_PROGRESS,
            EventType.MISSING_PRECONDITION,
            EventType.BUDGET_PRESSURE,
        ),
        max_extra_model_calls=1,
        max_extra_env_steps=0,
        max_added_tokens=192,
        read_only=True,
    ),
}


@dataclass(frozen=True)
class RequestContext:
    remaining_steps: int
    non_none_used: bool = False
    last_action_rejected: bool = False
    last_action_repeated: bool = False


@dataclass(frozen=True)
class ActionDecision:
    requested: HarnessAction
    selected: HarnessAction
    allowed: bool
    reason: str
    contract_version: str


def resolve_action_request(
    requested: HarnessAction,
    event: EventType,
    context: RequestContext,
) -> ActionDecision:
    """Apply action masks and deterministic NONE fallback."""

    contract = ACTION_CONTRACTS[requested]
    if requested is HarnessAction.NONE:
        return ActionDecision(requested, requested, True, "baseline", contract.version)

    if context.non_none_used:
        return ActionDecision(
            requested, HarnessAction.NONE, False, "intervention_budget_spent", contract.version
        )
    if context.remaining_steps <= 0:
        return ActionDecision(
            requested, HarnessAction.NONE, False, "no_remaining_steps", contract.version
        )
    if event not in contract.allowed_events:
        return ActionDecision(
            requested, HarnessAction.NONE, False, "event_incompatible", contract.version
        )
    if requested is HarnessAction.RETRY:
        retry_supported = (
            event is EventType.INVALID_REJECTED and context.last_action_rejected
        ) or (event is EventType.NO_PROGRESS and context.last_action_repeated)
        if not retry_supported:
            return ActionDecision(
                requested, HarnessAction.NONE, False, "retry_evidence_missing", contract.version
            )

    return ActionDecision(requested, requested, True, "allowed", contract.version)

