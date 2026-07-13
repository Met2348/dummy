from __future__ import annotations

from traceh_core.baseline import GenerationResult, run_baseline_episode


class FakeEnv:
    def __init__(self) -> None:
        self.actions: list[str] = []

    def reset(self):
        return ["Put the egg on the counter.\nYou are in a kitchen."], {
            "extra.gamefile": ["fake/game.tw-pddl"],
            "admissible_commands": [["look", "go to cabinet 1"]],
            "won": [False],
        }

    def step(self, actions: list[str]):
        self.actions.extend(actions)
        if len(self.actions) == 1:
            return ["You are at cabinet 1."], [0.0], [False], {
                "extra.gamefile": ["fake/game.tw-pddl"],
                "admissible_commands": [["look", "open cabinet 1"]],
                "won": [False],
            }
        return ["The cabinet is visible."], [0.0], [True], {
            "extra.gamefile": ["fake/game.tw-pddl"],
            "admissible_commands": [["look"]],
            "won": [False],
        }


def test_episode_runner_records_parser_failure_without_silent_model_retry() -> None:
    env = FakeEnv()
    outputs = iter(
        [
            GenerationResult("ACTION: go to cabinet 1", 100, 6),
            GenerationResult("I cannot decide.", 110, 5),
        ]
    )
    calls = 0

    def generate(_messages, _commands):
        nonlocal calls
        calls += 1
        return next(outputs)

    result = run_baseline_episode(env, generate=generate, max_steps=5)

    assert calls == 2
    assert env.actions == ["go to cabinet 1", "look"]
    assert result["task_id"] == "fake/game.tw-pddl"
    assert result["steps"] == 2
    assert result["input_tokens"] == 210
    assert result["output_tokens"] == 11
    assert result["invalid_actions"] == 1
    assert result["parser_failure"] is True
    assert result["success"] is False
    assert result["termination_reason"] == "environment_done"
    assert len(result["trace"]) == 2
    assert result["trace"][0]["parser_matched"] is True
    assert result["trace"][1]["parser_reason"] == "missing_action_line"
    assert len(result["trace"][0]["prefix_state_hash"]) == 64


def test_episode_runner_supports_indexed_admissible_action_protocol() -> None:
    env = FakeEnv()
    outputs = iter(
        [
            GenerationResult("ACTION_ID: 0", 90, 5),
            GenerationResult("ACTION_ID: 1", 95, 5),
        ]
    )

    result = run_baseline_episode(
        env,
        generate=lambda _messages, _commands: next(outputs),
        max_steps=2,
        action_protocol="indexed-v1",
    )

    assert env.actions == ["go to cabinet 1", "open cabinet 1"]
    assert result["invalid_actions"] == 0
    assert result["trace"][0]["parser_reason"] == "valid_action_id"


def test_episode_runner_supports_normalized_indexed_protocol() -> None:
    env = FakeEnv()
    outputs = iter(
        [
            GenerationResult("0", 90, 1),
            GenerationResult("1: 1", 95, 3),
        ]
    )
    result = run_baseline_episode(
        env,
        generate=lambda _messages, _commands: next(outputs),
        max_steps=2,
        action_protocol="indexed-normalized-v2",
    )
    assert env.actions == ["go to cabinet 1", "open cabinet 1"]
    assert result["invalid_actions"] == 0


def test_episode_runner_supports_single_token_label_protocol() -> None:
    env = FakeEnv()
    outputs = iter(
        [
            GenerationResult("ACTION_LABEL: A", 90, 1),
            GenerationResult("ACTION_LABEL: B", 95, 1),
        ]
    )
    seen_commands = []

    def generate(_messages, commands):
        seen_commands.append(list(commands))
        return next(outputs)

    result = run_baseline_episode(
        env,
        generate=generate,
        max_steps=2,
        action_protocol="label-logit-v1",
    )
    assert seen_commands == [
        ["go to cabinet 1", "look"],
        ["look", "open cabinet 1"],
    ]
    assert env.actions == ["go to cabinet 1", "open cabinet 1"]
    assert result["invalid_actions"] == 0
