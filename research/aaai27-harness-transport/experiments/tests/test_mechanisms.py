from __future__ import annotations

from traceh_core.mechanisms import (
    MechanismVariant,
    PrefixContext,
    command_scores,
    content_terms,
    propose_mechanism_action,
    proposal_table,
    same_state_action_counts,
    task_goal_text,
)


def context(
    *,
    task: str = "Put the apple on the table.",
    observation: str = "Kitchen.\nYou see apple 1 on counter 1.",
    history: tuple[tuple[str, str], ...] = (),
    commands: tuple[str, ...] = ("look", "take apple 1 from counter 1", "go to table 1"),
) -> PrefixContext:
    return PrefixContext(
        task=task,
        observation=observation,
        history=history,
        admissible_commands=commands,
    )


def test_content_terms_remove_protocol_stop_words() -> None:
    assert "apple" in content_terms("Put the apple on the table.")
    assert "put" not in content_terms("Put the apple on the table.")


def test_task_goal_text_ignores_initial_room_listing() -> None:
    text = "You see armchair 1 and drawer 1.\n\nYour task is to: put two newspaper in drawer."
    assert task_goal_text(text).strip() == "put two newspaper in drawer."


def test_same_state_action_counts_detect_visible_loop() -> None:
    prefix = context(
        task="Task.\nRoom A.",
        observation="Room A.",
        history=(
            ("go to room b", "Room B."),
            ("go to room a", "Room A."),
            ("go to room b", "Room B."),
            ("go to room a", "Room A."),
        ),
        commands=("look", "go to room b"),
    )
    counts = same_state_action_counts(prefix)
    assert counts["go to room b"] == 2


def test_anti_loop_retry_prefers_non_repeated_escape_action() -> None:
    prefix = context(
        task="Find the apple.",
        observation="Room A.",
        history=(
            ("go to room b", "Room B."),
            ("go to room a", "Room A."),
            ("go to room b", "Room B."),
            ("go to room a", "Room A."),
        ),
        commands=("look", "go to room b", "open drawer 1"),
    )
    proposal = propose_mechanism_action(prefix, MechanismVariant.ANTI_LOOP_RETRY)
    assert proposal.intervene
    assert proposal.selected_command == "open drawer 1"
    assert "escape_repeated_state_action" in proposal.reasons


def test_precondition_check_takes_visible_task_object_before_putting() -> None:
    prefix = context(
        commands=(
            "look",
            "put apple 1 on table 1",
            "take apple 1 from counter 1",
        )
    )
    proposal = propose_mechanism_action(prefix, MechanismVariant.PRECONDITION_CHECK)
    assert proposal.intervene
    assert proposal.selected_command == "take apple 1 from counter 1"
    assert "take_visible_task_object_before_delivery" in proposal.reasons


def test_precondition_check_treats_move_as_delivery_action() -> None:
    prefix = context(
        task="Your task is to: put two newspaper in drawer.",
        observation="You are carrying: newspaper 2.",
        commands=(
            "look",
            "move newspaper 2 to drawer 10",
            "go to drawer 11",
        ),
    )
    proposal = propose_mechanism_action(prefix, MechanismVariant.PRECONDITION_CHECK)
    assert proposal.intervene
    assert proposal.selected_command == "move newspaper 2 to drawer 10"
    assert "deliver_carried_task_object" in proposal.reasons


def test_subgoal_ledger_prefers_undelivered_second_object() -> None:
    prefix = context(
        task="Put two pens on the desk.",
        observation="Office.\nYou see pen 2 on shelf 1.",
        history=(("put pen 1 on desk 1", "Office.\nPen 1 is on desk 1."),),
        commands=("look", "take pen 1 from desk 1", "take pen 2 from shelf 1"),
    )
    proposal = propose_mechanism_action(prefix, MechanismVariant.SUBGOAL_LEDGER)
    assert proposal.intervene
    assert proposal.selected_command == "take pen 2 from shelf 1"


def test_bundle_abstains_when_no_visible_support() -> None:
    prefix = context(
        task="Solve the task.",
        observation="A plain room.",
        history=(),
        commands=("look", "wait", "inventory"),
    )
    proposal = propose_mechanism_action(prefix, MechanismVariant.BUNDLE_CONSERVATIVE)
    assert not proposal.intervene
    assert proposal.selected_command is None
    assert "abstain_low_confidence" in proposal.reasons


def test_bundle_uses_goal_not_room_listing_for_navigation() -> None:
    prefix = context(
        task=(
            "You see armchair 1, cabinet 1, drawer 1, and sofa 1.\n\n"
            "Your task is to: put two newspaper in drawer."
        ),
        observation="You are in the middle of a room. Looking quickly around you, you see nothing.",
        history=(("look", "You are in the middle of a room. Looking quickly around you, you see nothing."),) * 3,
        commands=("go to armchair 1", "go to drawer 1", "go to sofa 1", "look"),
    )
    proposal = propose_mechanism_action(prefix, MechanismVariant.BUNDLE_CONSERVATIVE)
    assert proposal.intervene
    assert proposal.selected_command == "go to drawer 1"


def test_command_scores_are_deterministic_and_explainable() -> None:
    prefix = context()
    first = command_scores(prefix, MechanismVariant.BUNDLE_CONSERVATIVE)
    second = command_scores(prefix, MechanismVariant.BUNDLE_CONSERVATIVE)
    assert first == second
    assert first[0][0] == "take apple 1 from counter 1"
    assert first[0][2]


def test_proposal_table_uses_serializable_rows() -> None:
    rows = proposal_table(
        context(),
        (MechanismVariant.PRECONDITION_CHECK, MechanismVariant.BUNDLE_CONSERVATIVE),
    )
    assert rows[0]["variant"] == "precondition_check"
    assert isinstance(rows[0]["reasons"], list)
