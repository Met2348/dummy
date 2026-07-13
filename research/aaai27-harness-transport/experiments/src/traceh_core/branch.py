"""Counterfactual branch prompting and continuation execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .agent import build_action_messages, parse_admissible_action
from .baseline import GenerationResult, _first, _normalized_info, _snapshot, _task_id
from .state import state_hash


def build_replan_messages(
    *,
    task: str,
    observation: str,
    history: Sequence[tuple[str, str]],
    admissible_commands: Sequence[str],
) -> list[dict[str, str]]:
    """Request a visible-state plan of no more than three future steps."""

    history_blocks = [
        f"Previous action: {action}\nResulting observation: {result}"
        for action, result in history[-4:]
    ]
    history_text = "\n\n".join(history_blocks) if history_blocks else "(none)"
    commands = "\n".join(f"- {command}" for command in sorted(admissible_commands))
    return [
        {
            "role": "system",
            "content": (
                "You are replanning for a text-environment agent using only the visible "
                "task, history, observation, and admissible commands."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task:\n{task}\n\n"
                f"Recent history:\n{history_text}\n\n"
                f"Current observation:\n{observation}\n\n"
                f"Currently admissible commands:\n{commands}\n\n"
                "Write a short plan of at most three numbered future steps. Do not claim "
                "to have observed or executed anything beyond the visible state."
            ),
        },
    ]


def run_episode_continuation(
    env: Any,
    *,
    task: str,
    observation: str,
    info: Mapping[str, Any],
    history: Sequence[tuple[str, str]],
    score: float,
    start_step: int,
    max_total_steps: int,
    generate: Callable[[list[dict[str, str]], Sequence[str]], GenerationResult],
    guidance: str | None = None,
    history_limit: int = 4,
) -> dict[str, Any]:
    """Continue one restored episode without resetting or adding hidden environment steps."""

    current_info = dict(info)
    current_observation = str(observation)
    current_history = list(history)
    current_score = float(score)
    success = bool(current_info.get("won", current_score > 0.0))
    trace: list[dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0
    invalid_actions = 0
    parser_failure = False
    termination_reason = "max_steps"

    for step_index in range(start_step, max_total_steps):
        commands = sorted(str(item) for item in current_info.get("admissible_commands", []))
        if not commands:
            termination_reason = "no_admissible_commands"
            break
        prefix_state = _snapshot(
            current_observation,
            current_info,
            step_index,
            current_score,
        )
        messages = build_action_messages(
            task=task,
            observation=current_observation,
            history=current_history[-history_limit:],
            admissible_commands=commands,
            guidance=guidance,
        )
        generation = generate(messages, commands)
        input_tokens += int(generation.input_tokens)
        output_tokens += int(generation.output_tokens)
        parsed = parse_admissible_action(generation.raw_output, commands)
        if not parsed.matched:
            invalid_actions += 1
            parser_failure = True

        next_observations, scores, dones, next_info_batch = env.step([parsed.action])
        next_observation = str(_first(next_observations))
        current_score = float(_first(scores))
        done = bool(_first(dones))
        next_info = _normalized_info(next_info_batch)
        success = bool(next_info.get("won", current_score > 0.0))
        next_state = _snapshot(
            next_observation,
            next_info,
            step_index + 1,
            current_score,
        )
        trace.append(
            {
                "step_index": step_index,
                "prefix_state": prefix_state,
                "prefix_state_hash": state_hash(prefix_state),
                "raw_output": generation.raw_output,
                "parser_candidate": parsed.candidate,
                "parser_matched": parsed.matched,
                "parser_reason": parsed.reason,
                "selected_action": parsed.action,
                "next_observation": next_observation,
                "next_state_hash": state_hash(next_state),
                "score": current_score,
                "done": done,
            }
        )
        current_history.append((parsed.action, next_observation))
        current_observation = next_observation
        current_info = next_info
        if done:
            termination_reason = "environment_done"
            break

    return {
        "task_id": _task_id(current_info),
        "success": success,
        "score": current_score,
        "steps": len(trace),
        "model_calls": len(trace),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "invalid_actions": invalid_actions,
        "parser_failure": parser_failure,
        "termination_reason": termination_reason,
        "trace": trace,
    }

