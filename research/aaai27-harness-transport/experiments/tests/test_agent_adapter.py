from __future__ import annotations

from traceh_core.agent import (
    build_action_messages,
    build_indexed_action_messages,
    build_labeled_action_messages,
    parse_admissible_action,
    parse_admissible_action_id,
    parse_normalized_action_id,
    parse_action_label,
    allowed_next_tokens,
)


COMMANDS = ["look", "go to cabinet 1", "open cabinet 1"]


def test_parser_accepts_one_exact_action_line_after_thinking_text() -> None:
    raw = "<think>I should inspect the room.</think>\nACTION: go to cabinet 1"
    result = parse_admissible_action(raw, COMMANDS)
    assert result.action == "go to cabinet 1"
    assert result.matched is True
    assert result.reason == "exact_action_line"


def test_parser_normalizes_case_and_whitespace_only() -> None:
    result = parse_admissible_action("ACTION:   OPEN   CABINET 1  ", COMMANDS)
    assert result.action == "open cabinet 1"
    assert result.matched is True


def test_parser_rejects_multiple_action_lines_without_silent_retry() -> None:
    result = parse_admissible_action(
        "ACTION: look\nACTION: open cabinet 1",
        COMMANDS,
    )
    assert result.action == "look"
    assert result.matched is False
    assert result.reason == "multiple_action_lines"


def test_parser_rejects_inadmissible_action_and_uses_deterministic_fallback() -> None:
    result = parse_admissible_action("ACTION: take egg 1", COMMANDS)
    assert result.action == "look"
    assert result.matched is False
    assert result.reason == "inadmissible_action"
    assert result.candidate == "take egg 1"


def test_parser_fallback_is_sorted_first_when_look_is_unavailable() -> None:
    result = parse_admissible_action(
        "I decline to choose.",
        ["open fridge 1", "go to countertop 1"],
    )
    assert result.action == "go to countertop 1"
    assert result.matched is False
    assert result.reason == "missing_action_line"


def test_prompt_exposes_only_task_history_observation_and_admissible_commands() -> None:
    messages = build_action_messages(
        task="Put the egg on the counter.",
        observation="You are in a kitchen.",
        history=[("look", "You see a cabinet.")],
        admissible_commands=COMMANDS,
    )
    assert [message["role"] for message in messages] == ["system", "user"]
    user = messages[1]["content"]
    assert "Put the egg on the counter." in user
    assert "Previous action: look" in user
    assert "Current observation:\nYou are in a kitchen." in user
    assert "- go to cabinet 1" in user
    assert "ACTION: <exact command>" in user


def test_prompt_includes_explicit_harness_guidance_when_provided() -> None:
    messages = build_action_messages(
        task="Put the egg on the counter.",
        observation="You are in a kitchen.",
        history=[],
        admissible_commands=COMMANDS,
        guidance="1. Find the egg.\n2. Heat the egg.",
    )
    assert "Harness guidance:" in messages[1]["content"]
    assert "1. Find the egg." in messages[1]["content"]


def test_indexed_parser_selects_command_by_zero_based_id() -> None:
    result = parse_admissible_action_id("ACTION_ID: 1", COMMANDS)
    assert result.action == "go to cabinet 1"
    assert result.matched is True
    assert result.reason == "valid_action_id"
    assert result.candidate == "1"


def test_indexed_parser_rejects_out_of_range_id() -> None:
    result = parse_admissible_action_id("ACTION_ID: 99", COMMANDS)
    assert result.action == "look"
    assert result.matched is False
    assert result.reason == "action_id_out_of_range"


def test_indexed_prompt_has_stable_ids_and_exact_output_contract() -> None:
    messages = build_indexed_action_messages(
        task="Put the egg on the counter.",
        observation="You are in a kitchen.",
        history=[],
        admissible_commands=COMMANDS,
    )
    user = messages[1]["content"]
    assert "[0] look" in user
    assert "[1] go to cabinet 1" in user
    assert "ACTION_ID: <integer>" in user


def test_normalized_indexed_parser_accepts_one_unique_bare_id() -> None:
    commands = [f"command {index}" for index in range(40)]
    result = parse_normalized_action_id("27", commands)
    assert result.action == "command 27"
    assert result.matched is True
    assert result.reason == "unique_action_id"


def test_normalized_indexed_parser_accepts_repeated_identical_id() -> None:
    commands = [f"command {index}" for index in range(40)]
    result = parse_normalized_action_id("32: 32", commands)
    assert result.action == "command 32"
    assert result.matched is True


def test_normalized_indexed_parser_rejects_conflicting_ids() -> None:
    commands = [f"command {index}" for index in range(40)]
    result = parse_normalized_action_id("32: 31", commands)
    assert result.matched is False
    assert result.reason == "conflicting_action_ids"


def test_label_parser_maps_one_exact_label_to_command() -> None:
    result = parse_action_label("ACTION_LABEL: B", COMMANDS)
    assert result.action == "go to cabinet 1"
    assert result.matched is True
    assert result.reason == "valid_action_label"


def test_labeled_prompt_assigns_stable_single_character_labels() -> None:
    messages = build_labeled_action_messages(
        task="Put the egg on the counter.",
        observation="You are in a kitchen.",
        history=[],
        admissible_commands=COMMANDS,
    )
    user = messages[1]["content"]
    assert "[A] look" in user
    assert "[B] go to cabinet 1" in user
    assert "ACTION_LABEL: <label>" in user


def test_command_trie_returns_only_tokens_on_valid_completion_paths() -> None:
    completions = [[10, 20], [10, 30], [40]]
    assert allowed_next_tokens(completions, [], eos_token_id=99) == [10, 40]
    assert allowed_next_tokens(completions, [10], eos_token_id=99) == [20, 30]
    assert allowed_next_tokens(completions, [10, 20], eos_token_id=99) == [99]


def test_command_trie_rejects_prefix_outside_all_completions() -> None:
    try:
        allowed_next_tokens([[10, 20]], [11], eos_token_id=99)
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("invalid trie prefix must fail")
