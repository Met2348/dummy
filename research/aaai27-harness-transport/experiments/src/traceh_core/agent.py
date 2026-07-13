"""Deterministic prompting and admissible-action parsing for agent runners."""

from __future__ import annotations

import re
import string
from collections.abc import Sequence
from dataclasses import dataclass


ACTION_LINE_RE = re.compile(r"^\s*ACTION\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
ACTION_ID_LINE_RE = re.compile(
    r"^\s*ACTION_ID\s*:\s*([+-]?\d+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
ACTION_LABELS = tuple(string.ascii_uppercase + string.ascii_lowercase)
ACTION_LABEL_LINE_RE = re.compile(r"^\s*ACTION_LABEL\s*:\s*([A-Za-z])\s*$", re.MULTILINE)


@dataclass(frozen=True)
class ActionParseResult:
    action: str
    matched: bool
    reason: str
    candidate: str | None


def _normalized_action(value: str) -> str:
    return " ".join(value.split()).casefold()


def _fallback_action(admissible_commands: Sequence[str]) -> str:
    commands = [str(command) for command in admissible_commands]
    if not commands:
        raise ValueError("admissible_commands cannot be empty")
    for command in commands:
        if _normalized_action(command) == "look":
            return command
    return min(commands, key=lambda command: (_normalized_action(command), command))


def allowed_next_tokens(
    completions: Sequence[Sequence[int]],
    generated_prefix: Sequence[int],
    *,
    eos_token_id: int,
) -> list[int]:
    """Return the token-trie frontier for a generated completion prefix."""

    prefix = list(generated_prefix)
    matching = [
        list(completion)
        for completion in completions
        if list(completion[: len(prefix)]) == prefix
    ]
    if not matching:
        raise ValueError("generated prefix is outside all admissible completion paths")
    frontier = {
        completion[len(prefix)] if len(completion) > len(prefix) else eos_token_id
        for completion in matching
    }
    return sorted(frontier)


def parse_admissible_action(
    raw_output: str,
    admissible_commands: Sequence[str],
) -> ActionParseResult:
    """Parse exactly one ACTION line or return a visible deterministic fallback."""

    commands = [str(command) for command in admissible_commands]
    fallback = _fallback_action(commands)
    candidates = ACTION_LINE_RE.findall(str(raw_output))
    if not candidates:
        return ActionParseResult(fallback, False, "missing_action_line", None)
    if len(candidates) != 1:
        return ActionParseResult(fallback, False, "multiple_action_lines", None)

    candidate = candidates[0].strip()
    normalized = _normalized_action(candidate)
    matches = [command for command in commands if _normalized_action(command) == normalized]
    if len(matches) != 1:
        return ActionParseResult(fallback, False, "inadmissible_action", candidate)
    return ActionParseResult(matches[0], True, "exact_action_line", candidate)


def parse_admissible_action_id(
    raw_output: str,
    admissible_commands: Sequence[str],
) -> ActionParseResult:
    """Map one zero-based ACTION_ID line to the corresponding command."""

    commands = [str(command) for command in admissible_commands]
    fallback = _fallback_action(commands)
    candidates = ACTION_ID_LINE_RE.findall(str(raw_output))
    if not candidates:
        return ActionParseResult(fallback, False, "missing_action_id_line", None)
    if len(candidates) != 1:
        return ActionParseResult(fallback, False, "multiple_action_id_lines", None)
    candidate = candidates[0]
    action_id = int(candidate)
    if action_id < 0 or action_id >= len(commands):
        return ActionParseResult(fallback, False, "action_id_out_of_range", candidate)
    return ActionParseResult(commands[action_id], True, "valid_action_id", candidate)


def parse_normalized_action_id(
    raw_output: str,
    admissible_commands: Sequence[str],
) -> ActionParseResult:
    """Accept an unambiguous integer ID without inferring or changing its value."""

    commands = [str(command) for command in admissible_commands]
    fallback = _fallback_action(commands)
    body = str(raw_output).strip()
    body = re.sub(r"^ACTION_ID\s*:\s*", "", body, flags=re.IGNORECASE)
    if not re.fullmatch(r"[+-]?\d+(?:\s*:\s*[+-]?\d+)*", body):
        return ActionParseResult(fallback, False, "invalid_action_id_format", None)
    values = [int(value) for value in re.findall(r"[+-]?\d+", body)]
    unique_values = set(values)
    if len(unique_values) != 1:
        return ActionParseResult(fallback, False, "conflicting_action_ids", body)
    action_id = values[0]
    if action_id < 0 or action_id >= len(commands):
        return ActionParseResult(fallback, False, "action_id_out_of_range", str(action_id))
    return ActionParseResult(commands[action_id], True, "unique_action_id", str(action_id))


def parse_action_label(
    raw_output: str,
    admissible_commands: Sequence[str],
) -> ActionParseResult:
    """Map one exact single-token label to its command."""

    commands = [str(command) for command in admissible_commands]
    fallback = _fallback_action(commands)
    if len(commands) > len(ACTION_LABELS):
        raise ValueError("label protocol supports at most 52 admissible commands")
    candidates = ACTION_LABEL_LINE_RE.findall(str(raw_output))
    if not candidates:
        return ActionParseResult(fallback, False, "missing_action_label_line", None)
    if len(candidates) != 1:
        return ActionParseResult(fallback, False, "multiple_action_label_lines", None)
    label = candidates[0]
    index = ACTION_LABELS.index(label)
    if index >= len(commands):
        return ActionParseResult(fallback, False, "action_label_out_of_range", label)
    return ActionParseResult(commands[index], True, "valid_action_label", label)


def build_action_messages(
    *,
    task: str,
    observation: str,
    history: Sequence[tuple[str, str]],
    admissible_commands: Sequence[str],
    guidance: str | None = None,
) -> list[dict[str, str]]:
    """Build a baseline prompt from information exposed by the environment."""

    if not admissible_commands:
        raise ValueError("admissible_commands cannot be empty")
    history_blocks = [
        f"Previous action: {action}\nResulting observation: {result}"
        for action, result in history
    ]
    history_text = "\n\n".join(history_blocks) if history_blocks else "(none)"
    command_text = "\n".join(f"- {command}" for command in sorted(admissible_commands))
    guidance_text = f"Harness guidance:\n{guidance}\n\n" if guidance else ""
    user = (
        f"Task:\n{task}\n\n"
        f"Recent history:\n{history_text}\n\n"
        f"Current observation:\n{observation}\n\n"
        f"{guidance_text}"
        f"Admissible commands:\n{command_text}\n\n"
        "Respond with exactly one line in this form:\nACTION: <exact command>"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a text-environment agent. Select exactly one command from the "
                "provided admissible-command list. Do not invent commands or add explanation."
            ),
        },
        {"role": "user", "content": user},
    ]



def build_indexed_action_messages(
    *,
    task: str,
    observation: str,
    history: Sequence[tuple[str, str]],
    admissible_commands: Sequence[str],
) -> list[dict[str, str]]:
    """Build a prompt that selects a command through its zero-based list ID."""

    if not admissible_commands:
        raise ValueError("admissible_commands cannot be empty")
    history_blocks = [
        f"Previous action: {action}\nResulting observation: {result}"
        for action, result in history
    ]
    history_text = "\n\n".join(history_blocks) if history_blocks else "(none)"
    command_text = "\n".join(
        f"[{index}] {command}" for index, command in enumerate(admissible_commands)
    )
    user = (
        f"Task:\n{task}\n\n"
        f"Recent history:\n{history_text}\n\n"
        f"Current observation:\n{observation}\n\n"
        f"Admissible commands with zero-based IDs:\n{command_text}\n\n"
        "Respond with exactly one line in this form:\nACTION_ID: <integer>"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a text-environment agent. Select exactly one zero-based ID from "
                "the provided admissible-command list. Do not add explanation."
            ),
        },
        {"role": "user", "content": user},
    ]


def build_labeled_action_messages(
    *,
    task: str,
    observation: str,
    history: Sequence[tuple[str, str]],
    admissible_commands: Sequence[str],
) -> list[dict[str, str]]:
    """Build a prompt whose command labels are single Qwen tokenizer tokens."""

    if not admissible_commands:
        raise ValueError("admissible_commands cannot be empty")
    if len(admissible_commands) > len(ACTION_LABELS):
        raise ValueError("label protocol supports at most 52 admissible commands")
    history_blocks = [
        f"Previous action: {action}\nResulting observation: {result}"
        for action, result in history
    ]
    history_text = "\n\n".join(history_blocks) if history_blocks else "(none)"
    command_text = "\n".join(
        f"[{ACTION_LABELS[index]}] {command}"
        for index, command in enumerate(admissible_commands)
    )
    user = (
        f"Task:\n{task}\n\n"
        f"Recent history:\n{history_text}\n\n"
        f"Current observation:\n{observation}\n\n"
        f"Admissible commands with labels:\n{command_text}\n\n"
        "Respond with exactly one line in this form:\nACTION_LABEL: <label>"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a text-environment agent. Select exactly one label from the "
                "provided admissible-command list. Do not add explanation."
            ),
        },
        {"role": "user", "content": user},
    ]
