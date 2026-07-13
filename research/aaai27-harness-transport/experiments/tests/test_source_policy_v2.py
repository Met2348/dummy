from __future__ import annotations

from traceh_core.source_policy_v2 import (
    SourcePolicyDecision,
    anti_loop_commands,
    build_deliberation_messages,
    evaluate_source_policy_gate,
    run_source_policy_continuation,
    run_source_policy_episode,
)


COMMANDS = ["look", "go to cabinet 1", "open cabinet 1"]


class FakeEnv:
    def __init__(self) -> None:
        self.actions: list[str] = []

    def reset(self):
        return ["Put the egg on the counter.\nYou are in a kitchen."], {
            "extra.gamefile": ["fake/game.tw-pddl"],
            "extra.expert_plan": [["forbidden oracle action"]],
            "admissible_commands": [["look", "go to cabinet 1"]],
            "won": [False],
        }

    def step(self, actions: list[str]):
        self.actions.extend(actions)
        if len(self.actions) == 1:
            return ["You are at cabinet 1."], [0.0], [False], {
                "extra.gamefile": ["fake/game.tw-pddl"],
                "extra.expert_plan": [["another forbidden action"]],
                "admissible_commands": [["look", "open cabinet 1"]],
                "won": [False],
            }
        return ["You solved the task."], [1.0], [True], {
            "extra.gamefile": ["fake/game.tw-pddl"],
            "extra.expert_plan": [[]],
            "admissible_commands": [["look"]],
            "won": [True],
        }


def test_anti_loop_blocks_action_repeated_twice_at_same_observation() -> None:
    task = "Task.\nRoom A."
    history = [
        ("go to room b", "Room B."),
        ("go to room a", "Room A."),
        ("go to room b", "Room B."),
        ("go to room a", "Room A."),
    ]
    result = anti_loop_commands(
        task=task,
        observation="Room A.",
        history=history,
        admissible_commands=["go to room b", "look"],
    )
    assert result == ["look"]


def test_anti_loop_never_returns_an_empty_command_set() -> None:
    task = "Task.\nRoom A."
    history = [
        ("look", "Room A."),
        ("look", "Room A."),
    ]
    result = anti_loop_commands(
        task=task,
        observation="Room A.",
        history=history,
        admissible_commands=["look"],
    )
    assert result == ["look"]


def test_deliberation_prompt_contains_memory_and_generic_recipe_without_oracle() -> None:
    messages = build_deliberation_messages(
        task="Heat the egg and put it on the counter.",
        observation="You are at fridge 1.",
        history=[("open fridge 1", "The fridge is open. An egg 1 is inside.")],
        admissible_commands=["take egg 1 from fridge 1", "close fridge 1"],
    )
    content = "\n".join(message["content"] for message in messages)
    assert "Current subgoal" in content
    assert "microwave" in content
    assert "open fridge 1" in content
    assert "take egg 1 from fridge 1" in content
    assert "expert_plan" not in content
    assert "forbidden oracle" not in content


def test_deliberation_prompt_can_ablate_persistent_memory() -> None:
    messages = build_deliberation_messages(
        task="Put a book on the sofa.",
        observation="You are at drawer 1.",
        history=[("open drawer 1", "The drawer is empty.")],
        admissible_commands=["go to drawer 2", "look"],
        memory_limit=0,
    )
    content = messages[1]["content"]
    assert "No transitions are exposed in this ablation." in content
    assert "open drawer 1" not in content


def test_source_policy_runner_records_two_calls_and_deliberation() -> None:
    env = FakeEnv()
    decisions = iter(
        [
            SourcePolicyDecision(
                raw_output="ACTION: go to cabinet 1",
                deliberation="Current subgoal: inspect the cabinet.",
                input_tokens=120,
                output_tokens=20,
                model_calls=2,
            ),
            SourcePolicyDecision(
                raw_output="ACTION: open cabinet 1",
                deliberation="Current subgoal: open the cabinet.",
                input_tokens=130,
                output_tokens=18,
                model_calls=2,
            ),
        ]
    )

    result = run_source_policy_episode(
        env,
        decide=lambda _messages, _commands: next(decisions),
        max_steps=5,
    )

    assert env.actions == ["go to cabinet 1", "open cabinet 1"]
    assert result["success"] is True
    assert result["score"] == 1.0
    assert result["steps"] == 2
    assert result["model_calls"] == 4
    assert result["input_tokens"] == 250
    assert result["output_tokens"] == 38
    assert result["trace"][0]["deliberation"].startswith("Current subgoal")
    assert result["trace"][0]["policy_admissible_commands"] == [
        "go to cabinet 1",
        "look",
    ]
    assert "expert_plan" not in str(result["trace"])


def test_source_policy_gate_reopens_branches_after_one_of_three_successes() -> None:
    result = evaluate_source_policy_gate(
        episode_count=3,
        success_count=1,
        parser_failure_rate=0.0,
        infrastructure_failure_count=0,
    )
    assert result["decision"] == "PASS_REOPEN_SOURCE_BRANCHES"
    assert result["passed"] is True


def test_source_policy_gate_stops_all_zero_scaffold() -> None:
    result = evaluate_source_policy_gate(
        episode_count=3,
        success_count=0,
        parser_failure_rate=0.0,
        infrastructure_failure_count=0,
    )
    assert result["decision"] == "STOP_SOURCE_POLICY_V2"
    assert result["passed"] is False


def test_source_policy_continuation_starts_at_prefix_and_injects_guidance() -> None:
    env = FakeEnv()
    env.actions = ["go to cabinet 1"]
    seen_messages: list[list[dict[str, str]]] = []

    def decide(messages, _commands):
        seen_messages.append(messages)
        return SourcePolicyDecision(
            raw_output="ACTION: open cabinet 1",
            deliberation="Current subgoal: follow the repaired plan.",
            input_tokens=100,
            output_tokens=15,
        )

    result = run_source_policy_continuation(
        env,
        task="Put the egg on the counter.\nYou are in a kitchen.",
        observation="You are at cabinet 1.",
        info={
            "extra.gamefile": "fake/game.tw-pddl",
            "admissible_commands": ["look", "open cabinet 1"],
            "won": False,
        },
        history=[("go to cabinet 1", "You are at cabinet 1.")],
        score=0.0,
        start_step=1,
        max_total_steps=5,
        decide=decide,
        guidance="Inspect the cabinet, then recover the object.",
    )
    assert env.actions == ["go to cabinet 1", "open cabinet 1"]
    assert result["trace"][0]["step_index"] == 1
    assert result["success"] is True
    assert "Harness REPLAN guidance" in seen_messages[0][1]["content"]
