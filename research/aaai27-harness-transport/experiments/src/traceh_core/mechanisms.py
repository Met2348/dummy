"""Structured harness-action mechanism variants for local design probes.

The functions here are deliberately model-agnostic.  They do not claim to solve
ALFWorld; they encode visible-state intervention proposals that can be tested in
synthetic branch-response probes before spending GPU budget.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum


class MechanismVariant(str, Enum):
    NONE = "none"
    NATURAL_REPLAN = "natural_replan"
    ANTI_LOOP_RETRY = "anti_loop_retry"
    PRECONDITION_CHECK = "precondition_check"
    SUBGOAL_LEDGER = "subgoal_ledger"
    BUNDLE_CONSERVATIVE = "bundle_conservative"


@dataclass(frozen=True)
class PrefixContext:
    task: str
    observation: str
    history: tuple[tuple[str, str], ...]
    admissible_commands: tuple[str, ...]
    event_type: str = "no_progress"


@dataclass(frozen=True)
class MechanismProposal:
    variant: MechanismVariant
    intervene: bool
    selected_command: str | None
    confidence: float
    reasons: tuple[str, ...]


STOP_WORDS = {
    "a",
    "an",
    "and",
    "at",
    "be",
    "both",
    "bring",
    "clean",
    "cool",
    "find",
    "from",
    "heat",
    "in",
    "into",
    "is",
    "it",
    "move",
    "of",
    "on",
    "one",
    "place",
    "put",
    "see",
    "take",
    "the",
    "then",
    "to",
    "two",
    "use",
    "you",
    "with",
}

ACTION_VERBS = ("take", "move", "put", "open", "close", "go", "use", "heat", "cool", "clean")
PASSIVE_COMMANDS = {"look", "inventory", "wait"}


def normalize_text(value: str) -> str:
    return " ".join(str(value).split()).casefold()


def content_terms(text: str) -> set[str]:
    tokens = re.findall(r"[a-z][a-z0-9_]*", normalize_text(text))
    terms = {token for token in tokens if token not in STOP_WORDS and len(token) > 1}
    singulars = {token[:-1] for token in terms if len(token) > 3 and token.endswith("s")}
    return terms | singulars


def task_goal_text(task: str) -> str:
    """Extract the explicit ALFWorld goal sentence when present."""

    text = str(task)
    marker = "your task is to:"
    lower = text.casefold()
    if marker not in lower:
        return text
    return text[lower.rfind(marker) + len(marker) :]


def _fallback_command(commands: Sequence[str]) -> str:
    if not commands:
        raise ValueError("admissible_commands cannot be empty")
    for command in commands:
        if normalize_text(command) == "look":
            return command
    return min(commands, key=lambda item: (normalize_text(item), item))


def _starts_action_verb(normalized_command: str) -> bool:
    return any(
        normalized_command == verb or normalized_command.startswith(f"{verb} ")
        for verb in ACTION_VERBS
    )


def _state_key(observation: str) -> str:
    return normalize_text(observation)


def same_state_action_counts(context: PrefixContext) -> Counter[str]:
    """Count previously selected actions at the current visible observation."""

    counts: Counter[str] = Counter()
    prior_states = [_initial_observation(context.task)] + [
        result for _action, result in context.history[:-1]
    ]
    for state, (action, _result) in zip(prior_states, context.history, strict=False):
        if _state_key(state) == _state_key(context.observation):
            counts[normalize_text(action)] += 1
    return counts


def _initial_observation(task: str) -> str:
    lines = str(task).splitlines()
    return "\n".join(lines[1:]) if len(lines) > 1 else str(task)


def visible_inventory_terms(observation: str) -> set[str]:
    lines = [line for line in str(observation).splitlines() if "carrying" in line.casefold()]
    return set().union(*(content_terms(line) for line in lines)) if lines else set()


def delivered_terms(history: Sequence[tuple[str, str]]) -> set[str]:
    delivered: set[str] = set()
    for action, _result in history:
        normalized = normalize_text(action)
        if normalized.startswith(("move ", "put ")):
            delivered.update(content_terms(action))
    return delivered


def delivered_object_phrases(history: Sequence[tuple[str, str]]) -> set[str]:
    delivered: set[str] = set()
    for action, _result in history:
        normalized = normalize_text(action)
        if not normalized.startswith(("move ", "put ")):
            continue
        match = re.search(r"\b(?:move|put)\s+([a-z][a-z0-9_]*)\s+([0-9]+)\b", normalized)
        if match:
            delivered.add(f"{match.group(1)} {match.group(2)}")
    return delivered


def command_scores(context: PrefixContext, variant: MechanismVariant) -> list[tuple[str, float, tuple[str, ...]]]:
    commands = [str(command) for command in context.admissible_commands]
    if not commands:
        raise ValueError("admissible_commands cannot be empty")
    task_terms = content_terms(task_goal_text(context.task))
    observation_terms = content_terms(context.observation)
    inventory_terms = visible_inventory_terms(context.observation)
    delivered = set() if "two" in normalize_text(context.task) else delivered_terms(context.history)
    delivered_objects = delivered_object_phrases(context.history)
    same_state_counts = same_state_action_counts(context)
    last_action = normalize_text(context.history[-1][0]) if context.history else ""
    rows: list[tuple[str, float, tuple[str, ...]]] = []

    for command in commands:
        normalized = normalize_text(command)
        terms = content_terms(command)
        score = 0.0
        reasons: list[str] = []

        if normalized in PASSIVE_COMMANDS:
            score -= 0.20
            reasons.append("avoid_passive_command")
        if normalized == last_action:
            score -= 0.15
            reasons.append("avoid_last_action")
        if same_state_counts[normalized] > 0:
            score -= 0.30 * same_state_counts[normalized]
            reasons.append("same_state_repeat")
        if normalized.startswith("take ") and any(phrase in normalized for phrase in delivered_objects):
            score -= 0.55
            reasons.append("avoid_already_delivered_instance")

        if variant is MechanismVariant.NATURAL_REPLAN:
            score += 0.10
            reasons.append("generic_change")
            if same_state_counts[normalized] == 0 and _starts_action_verb(normalized):
                score += 0.25
                reasons.append("non_repeated_candidate")

        elif variant is MechanismVariant.ANTI_LOOP_RETRY:
            if same_state_counts[normalized] == 0 and _starts_action_verb(normalized):
                score += 0.70
                reasons.append("escape_repeated_state_action")

        elif variant is MechanismVariant.PRECONDITION_CHECK:
            score += _precondition_bonus(normalized, terms, task_terms, observation_terms, inventory_terms, reasons)

        elif variant is MechanismVariant.SUBGOAL_LEDGER:
            score += _ledger_bonus(normalized, terms, task_terms, observation_terms, inventory_terms, delivered, reasons)

        elif variant is MechanismVariant.BUNDLE_CONSERVATIVE:
            score += _precondition_bonus(normalized, terms, task_terms, observation_terms, inventory_terms, reasons)
            score += _ledger_bonus(normalized, terms, task_terms, observation_terms, inventory_terms, delivered, reasons)
            if same_state_counts[normalized] == 0 and _starts_action_verb(normalized):
                score += 0.25
                reasons.append("anti_loop_support")

        rows.append((command, score, tuple(reasons)))

    return sorted(rows, key=lambda row: (-row[1], normalize_text(row[0]), row[0]))


def _precondition_bonus(
    normalized_command: str,
    command_terms: set[str],
    task_terms: set[str],
    observation_terms: set[str],
    inventory_terms: set[str],
    reasons: list[str],
) -> float:
    bonus = 0.0
    task_overlap = len(command_terms & task_terms)
    visible_overlap = len(command_terms & observation_terms)
    carrying = bool(inventory_terms)

    if normalized_command.startswith("take ") and not carrying and task_overlap:
        bonus += 0.85 + 0.05 * visible_overlap
        reasons.append("take_visible_task_object_before_delivery")
    if normalized_command.startswith("open ") and ("closed" in observation_terms or "microwave" in command_terms):
        bonus += 0.95
        reasons.append("open_container_precondition")
    if normalized_command.startswith(("move ", "put ")) and "closed" in observation_terms and "microwave" in command_terms:
        bonus -= 0.60
        reasons.append("blocked_by_closed_container")
    if normalized_command.startswith(("move ", "put ")) and carrying and task_overlap:
        bonus += 0.75
        reasons.append("deliver_carried_task_object")
    if normalized_command.startswith(("move ", "put ")) and not carrying:
        bonus -= 0.40
        reasons.append("cannot_put_without_inventory")
    if normalized_command.startswith("use ") and carrying:
        bonus += 0.25
        reasons.append("use_after_carrying")
    return bonus


def _ledger_bonus(
    normalized_command: str,
    command_terms: set[str],
    task_terms: set[str],
    observation_terms: set[str],
    inventory_terms: set[str],
    delivered: set[str],
    reasons: list[str],
) -> float:
    bonus = 0.0
    remaining_task_terms = task_terms - delivered
    target_overlap = len(command_terms & remaining_task_terms)
    visible_overlap = len(command_terms & observation_terms)

    if normalized_command.startswith("take ") and target_overlap and visible_overlap:
        bonus += 0.75
        reasons.append("advance_remaining_subgoal_object")
    if normalized_command.startswith(("move ", "put ")) and inventory_terms and target_overlap:
        bonus += 0.65
        reasons.append("finish_current_subgoal")
    if normalized_command.startswith("go to ") and target_overlap:
        bonus += 0.35
        reasons.append("navigate_toward_remaining_subgoal")
    if normalized_command.startswith("open ") and target_overlap:
        bonus += 0.30
        reasons.append("inspect_relevant_container")
    return bonus


def propose_mechanism_action(
    context: PrefixContext,
    variant: MechanismVariant,
    *,
    confidence_threshold: float | None = None,
) -> MechanismProposal:
    """Select a visible-state command proposal for one mechanism variant."""

    if variant is MechanismVariant.NONE:
        return MechanismProposal(variant, False, None, 0.0, ("baseline_none",))

    scores = command_scores(context, variant)
    selected, score, reasons = scores[0]
    second_score = scores[1][1] if len(scores) > 1 else 0.0
    margin = max(0.0, score - second_score)
    confidence = max(0.0, min(1.0, 0.50 * score + 0.25 * margin))
    threshold = _default_threshold(variant) if confidence_threshold is None else confidence_threshold
    intervene = confidence >= threshold and normalize_text(selected) != normalize_text(_fallback_command(context.admissible_commands))
    if not intervene:
        return MechanismProposal(variant, False, None, confidence, ("abstain_low_confidence",) + reasons)
    return MechanismProposal(variant, True, selected, confidence, reasons)


def _default_threshold(variant: MechanismVariant) -> float:
    if variant is MechanismVariant.NATURAL_REPLAN:
        return 0.10
    if variant is MechanismVariant.ANTI_LOOP_RETRY:
        return 0.25
    if variant is MechanismVariant.BUNDLE_CONSERVATIVE:
        return 0.20
    return 0.35


def mechanism_variants() -> tuple[MechanismVariant, ...]:
    return tuple(variant for variant in MechanismVariant if variant is not MechanismVariant.NONE)


def proposal_table(context: PrefixContext, variants: Iterable[MechanismVariant]) -> list[dict[str, object]]:
    rows = []
    for variant in variants:
        proposal = propose_mechanism_action(context, variant)
        rows.append(
            {
                "variant": proposal.variant.value,
                "intervene": proposal.intervene,
                "selected_command": proposal.selected_command,
                "confidence": proposal.confidence,
                "reasons": list(proposal.reasons),
            }
        )
    return rows
