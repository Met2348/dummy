"""Model-agnostic baseline episode execution for batched text environments."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .agent import (
    build_action_messages,
    build_indexed_action_messages,
    build_labeled_action_messages,
    parse_admissible_action,
    parse_admissible_action_id,
    parse_normalized_action_id,
    parse_action_label,
)
from .state import state_hash


@dataclass(frozen=True)
class GenerationResult:
    raw_output: str
    input_tokens: int
    output_tokens: int


def _first(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _normalized_info(info: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _first(value) for key, value in info.items()}


def _task_id(info: Mapping[str, Any]) -> str:
    for key in ("extra.gamefile", "gamefile", "extra.game_file"):
        if key in info:
            return str(info[key])
    return "unknown-game"


def _snapshot(
    observation: str,
    info: Mapping[str, Any],
    step_index: int,
    score: float,
) -> dict[str, Any]:
    commands = sorted(str(item) for item in info.get("admissible_commands", []))
    inventory = [line.strip() for line in observation.splitlines() if "carrying" in line.casefold()]
    return {
        "task_id": _task_id(info),
        "location": observation.splitlines()[0].strip() if observation else "",
        "inventory": inventory,
        "completed_predicates": [
            f"won={bool(info.get('won', False))}",
            f"score={score:.6f}",
        ],
        "admissible_commands": commands,
        "step_index": step_index,
        "recent_tool_result": {
            "observation": observation,
            "won": bool(info.get("won", False)),
        },
    }


def run_baseline_episode(
    env: Any,
    *,
    generate: Callable[[list[dict[str, str]], Sequence[str]], GenerationResult],
    max_steps: int,
    history_limit: int = 4,
    action_protocol: str = "exact-text-v1",
) -> dict[str, Any]:
    """Run one episode with one generation call per environment step."""

    observations, info_batch = env.reset()
    observation = str(_first(observations))
    info = _normalized_info(info_batch)
    task = observation
    task_id = _task_id(info)
    history: list[tuple[str, str]] = []
    trace: list[dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0
    invalid_actions = 0
    parser_failure = False
    score = 0.0
    success = bool(info.get("won", False))
    termination_reason = "max_steps"

    for step_index in range(max_steps):
        commands = sorted(str(item) for item in info.get("admissible_commands", []))
        if not commands:
            termination_reason = "no_admissible_commands"
            break
        prefix_state = _snapshot(observation, info, step_index, score)
        if action_protocol == "label-logit-v1":
            prompt_builder = build_labeled_action_messages
        elif action_protocol in {"indexed-v1", "indexed-normalized-v2"}:
            prompt_builder = build_indexed_action_messages
        else:
            prompt_builder = build_action_messages
        if action_protocol not in {
            "exact-text-v1",
            "indexed-v1",
            "indexed-normalized-v2",
            "label-logit-v1",
        }:
            raise ValueError(f"unknown action protocol: {action_protocol}")
        messages = prompt_builder(
            task=task,
            observation=observation,
            history=history[-history_limit:],
            admissible_commands=commands,
        )
        generation = generate(messages, commands)
        input_tokens += int(generation.input_tokens)
        output_tokens += int(generation.output_tokens)
        if action_protocol == "indexed-v1":
            parsed = parse_admissible_action_id(generation.raw_output, commands)
        elif action_protocol == "indexed-normalized-v2":
            parsed = parse_normalized_action_id(generation.raw_output, commands)
        elif action_protocol == "label-logit-v1":
            parsed = parse_action_label(generation.raw_output, commands)
        else:
            parsed = parse_admissible_action(generation.raw_output, commands)
        if not parsed.matched:
            invalid_actions += 1
            parser_failure = True

        next_observations, scores, dones, next_info_batch = env.step([parsed.action])
        next_observation = str(_first(next_observations))
        score = float(_first(scores))
        done = bool(_first(dones))
        next_info = _normalized_info(next_info_batch)
        success = bool(next_info.get("won", score > 0.0))
        next_state = _snapshot(next_observation, next_info, step_index + 1, score)
        trace.append(
            {
                "step_index": step_index,
                "action_protocol": action_protocol,
                "prefix_state": prefix_state,
                "prefix_state_hash": state_hash(prefix_state),
                "raw_output": generation.raw_output,
                "parser_candidate": parsed.candidate,
                "parser_matched": parsed.matched,
                "parser_reason": parsed.reason,
                "selected_action": parsed.action,
                "next_observation": next_observation,
                "next_state_hash": state_hash(next_state),
                "score": score,
                "done": done,
            }
        )
        history.append((parsed.action, next_observation))
        observation = next_observation
        info = next_info
        if done:
            termination_reason = "environment_done"
            break

    return {
        "task_id": task_id,
        "success": success,
        "score": score,
        "steps": len(trace),
        "model_calls": len(trace),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "invalid_actions": invalid_actions,
        "parser_failure": parser_failure,
        "termination_reason": termination_reason,
        "trace": trace,
    }
