from __future__ import annotations

from traceh_core.baseline import GenerationResult
from traceh_core.branch import build_replan_messages, run_episode_continuation


class ContinuationEnv:
    def __init__(self) -> None:
        self.actions: list[str] = []

    def step(self, actions: list[str]):
        self.actions.extend(actions)
        done = len(self.actions) == 2
        return [f"observation {len(self.actions)}"], [float(done)], [done], {
            "extra.gamefile": ["fake/game.tw-pddl"],
            "admissible_commands": [["look", "open cabinet 1"]],
            "won": [done],
        }


def test_replan_prompt_uses_visible_state_and_caps_plan_to_three_steps() -> None:
    messages = build_replan_messages(
        task="Put the egg on the counter.",
        observation="You face a cabinet.",
        history=[("go to cabinet 1", "You face a cabinet.")],
        admissible_commands=["look", "open cabinet 1"],
    )
    user = messages[1]["content"]
    assert "Put the egg on the counter." in user
    assert "open cabinet 1" in user
    assert "at most three" in user


def test_continuation_starts_at_prefix_and_injects_guidance_without_extra_env_step() -> None:
    env = ContinuationEnv()
    seen_messages = []
    outputs = iter(
        [
            GenerationResult("ACTION: open cabinet 1", 80, 6),
            GenerationResult("ACTION: look", 70, 4),
        ]
    )

    def generate(messages, _commands):
        seen_messages.append(messages)
        return next(outputs)

    result = run_episode_continuation(
        env,
        task="Put the egg on the counter.",
        observation="You face a cabinet.",
        info={
            "extra.gamefile": "fake/game.tw-pddl",
            "admissible_commands": ["look", "open cabinet 1"],
            "won": False,
        },
        history=[("go to cabinet 1", "You face a cabinet.")],
        score=0.0,
        start_step=7,
        max_total_steps=9,
        generate=generate,
        guidance="Open the cabinet, then inspect.",
    )
    assert env.actions == ["open cabinet 1", "look"]
    assert [item["step_index"] for item in result["trace"]] == [7, 8]
    assert result["steps"] == 2
    assert result["model_calls"] == 2
    assert result["success"] is True
    assert "Harness guidance:" in seen_messages[0][1]["content"]

