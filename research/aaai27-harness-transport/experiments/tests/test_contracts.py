from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from traceh_core.contracts import (
    ACTION_CONTRACTS,
    EventType,
    HarnessAction,
    RequestContext,
    resolve_action_request,
)
from traceh_core.schema import load_schema, validate_instance


SCHEMA = Path(__file__).parents[1] / "schemas" / "traceh-action-contract.schema.json"
VARIANTS = ("standard", "retry_evidence_missing", "intervention_spent", "zero_budget")
CASES = list(itertools.product(EventType, HarnessAction, VARIANTS))


def _context(event: EventType, variant: str) -> RequestContext:
    context = RequestContext(
        remaining_steps=10,
        last_action_rejected=event is EventType.INVALID_REJECTED,
        last_action_repeated=event is EventType.NO_PROGRESS,
    )
    if variant == "retry_evidence_missing":
        return RequestContext(remaining_steps=10)
    if variant == "intervention_spent":
        return RequestContext(
            remaining_steps=10,
            non_none_used=True,
            last_action_rejected=context.last_action_rejected,
            last_action_repeated=context.last_action_repeated,
        )
    if variant == "zero_budget":
        return RequestContext(
            remaining_steps=0,
            last_action_rejected=context.last_action_rejected,
            last_action_repeated=context.last_action_repeated,
        )
    return context


@pytest.mark.parametrize(("event", "action", "variant"), CASES)
def test_contract_matrix_has_deterministic_none_fallback(
    event: EventType, action: HarnessAction, variant: str
) -> None:
    decision = resolve_action_request(action, event, _context(event, variant))
    if decision.allowed:
        assert decision.selected is action
    else:
        assert decision.selected is HarnessAction.NONE
    if action is HarnessAction.NONE:
        assert decision.allowed
    if variant in {"intervention_spent", "zero_budget"} and action is not HarnessAction.NONE:
        assert not decision.allowed


def test_contract_matrix_is_exactly_80_cases() -> None:
    assert len(CASES) == 80


def test_contracts_validate_against_current_schema() -> None:
    schema = load_schema(SCHEMA)
    for contract in ACTION_CONTRACTS.values():
        validate_instance(contract.to_dict(), schema)


def test_action_budget_invariants() -> None:
    check = ACTION_CONTRACTS[HarnessAction.CHECK]
    retry = ACTION_CONTRACTS[HarnessAction.RETRY]
    replan = ACTION_CONTRACTS[HarnessAction.REPLAN]
    assert check.read_only and check.max_extra_env_steps == 0 and check.max_added_tokens <= 128
    assert retry.max_extra_env_steps == 1 and retry.max_extra_model_calls == 0
    assert replan.max_extra_model_calls == 1 and replan.max_added_tokens <= 192
    assert all(not contract.adds_oracle_information for contract in ACTION_CONTRACTS.values())

