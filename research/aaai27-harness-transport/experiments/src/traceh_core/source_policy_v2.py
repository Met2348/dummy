"""Auditable deliberation, memory, and anti-loop scaffold for source qualification."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .agent import parse_admissible_action
from .baseline import _first, _normalized_info, _snapshot, _task_id
from .state import state_hash


GENERIC_RECIPES = """Generic ALFWorld procedures:
- Pick/place: locate the object, take it, navigate to the destination, then put it there.
- Heat: take the object; use the microwave by opening it, putting the object in, closing it,
  turning it on, then retrieving the object before placing it at the destination.
- Cool: take the object, put it in the fridge, close the fridge, then retrieve and place it.
- Clean: take the object, put it in a sink basin, use the cleaning action, retrieve and place it.
- Examine under light: take the object, navigate to a lamp, and use the lamp as required.
- Two-object tasks: complete the delivery for both requested instances, one at a time.
These are generic procedures, not task-specific gold actions."""


@dataclass(frozen=True)
class SourcePolicyDecision:
    raw_output: str
    deliberation: str
    input_tokens: int
    output_tokens: int
    model_calls: int = 2


def evaluate_source_policy_gate(
    *,
    episode_count: int,
    success_count: int,
    parser_failure_rate: float,
    infrastructure_failure_count: int,
) -> dict[str, Any]:
    """Apply the frozen three-episode qualification gate."""

    protocol_ok = (
        episode_count == 3
        and infrastructure_failure_count == 0
        and parser_failure_rate <= 0.02
    )
    passed = protocol_ok and success_count >= 1
    if not protocol_ok:
        decision = "BLOCKED_SOURCE_POLICY_PROTOCOL"
    elif passed:
        decision = "PASS_REOPEN_SOURCE_BRANCHES"
    else:
        decision = "STOP_SOURCE_POLICY_V2"
    return {
        "decision": decision,
        "passed": passed,
        "protocol_ok": protocol_ok,
        "required_episode_count": 3,
        "required_success_count": 1,
        "parser_failure_threshold": 0.02,
    }


def _state_key(observation: str) -> str:
    return " ".join(str(observation).split()).casefold()


def _initial_observation(task: str) -> str:
    lines = str(task).splitlines()
    return "\n".join(lines[1:]) if len(lines) > 1 else str(task)


def anti_loop_commands(
    *,
    task: str,
    observation: str,
    history: Sequence[tuple[str, str]],
    admissible_commands: Sequence[str],
    repeat_limit: int = 2,
) -> list[str]:
    """Hide actions already taken repeatedly from the same visible state."""

    commands = [str(command) for command in admissible_commands]
    if not commands:
        raise ValueError("admissible_commands cannot be empty")
    if repeat_limit <= 0:
        raise ValueError("repeat_limit must be positive")

    prior_states = (
        [_initial_observation(task)] + [result for _, result in history[:-1]]
        if history
        else []
    )
    counts = Counter(
        (_state_key(state), str(action).casefold())
        for state, (action, _result) in zip(prior_states, history, strict=True)
    )
    current_key = _state_key(observation)
    filtered = [
        command
        for command in commands
        if counts[(current_key, command.casefold())] < repeat_limit
    ]
    return filtered or commands


def _memory_text(history: Sequence[tuple[str, str]], memory_limit: int) -> str:
    if memory_limit < 0:
        raise ValueError("memory_limit cannot be negative")
    if memory_limit == 0:
        return "No transitions are exposed in this ablation."
    if not history:
        return "No transitions have been executed yet."
    recent = history[-memory_limit:]
    blocks = [
        f"Step {len(history) - len(recent) + index}: action={action}\nresult={result}"
        for index, (action, result) in enumerate(recent)
    ]
    return "\n\n".join(blocks)


def build_deliberation_messages(
    *,
    task: str,
    observation: str,
    history: Sequence[tuple[str, str]],
    admissible_commands: Sequence[str],
    guidance: str | None = None,
    memory_limit: int = 12,
) -> list[dict[str, str]]:
    """Build the source-policy v2 planning prompt from visible information only."""

    commands = [str(command) for command in admissible_commands]
    if not commands:
        raise ValueError("admissible_commands cannot be empty")
    command_text = "\n".join(f"- {command}" for command in commands)
    guidance_text = (
        f"Harness REPLAN guidance:\n{guidance}\n\n" if guidance else ""
    )
    user = (
        f"Task:\n{task}\n\n"
        f"{GENERIC_RECIPES}\n\n"
        f"Persistent transition memory:\n{_memory_text(history, memory_limit)}\n\n"
        f"Current observation:\n{observation}\n\n"
        f"{guidance_text}"
        f"Currently selectable commands:\n{command_text}\n\n"
        "Write a compact decision note with exactly these labels:\n"
        "Current subgoal: ...\nKnown evidence: ...\nAvoid: ...\nBest command: ...\n"
        "The best command must be copied exactly from the selectable list."
    )
    return [
        {
            "role": "system",
            "content": (
                "You are planning for a text-environment agent. Use only the supplied task, "
                "visible transition memory, current observation, and selectable commands. "
                "Never assume hidden state or a gold plan."
            ),
        },
        {"role": "user", "content": user},
    ]


def run_source_policy_episode(
    env: Any,
    *,
    decide: Callable[[list[dict[str, str]], Sequence[str]], SourcePolicyDecision],
    max_steps: int,
    repeat_limit: int = 2,
    memory_limit: int = 12,
) -> dict[str, Any]:
    """Run one v2 source-policy episode with two model calls per environment step."""

    observations, info_batch = env.reset()
    observation = str(_first(observations))
    info = _normalized_info(info_batch)
    task = observation
    return run_source_policy_continuation(
        env,
        task=task,
        observation=observation,
        info=info,
        history=[],
        score=0.0,
        start_step=0,
        max_total_steps=max_steps,
        decide=decide,
        repeat_limit=repeat_limit,
        memory_limit=memory_limit,
    )


def run_source_policy_continuation(
    env: Any,
    *,
    task: str,
    observation: str,
    info: Mapping[str, Any],
    history: Sequence[tuple[str, str]],
    score: float,
    start_step: int,
    max_total_steps: int,
    decide: Callable[[list[dict[str, str]], Sequence[str]], SourcePolicyDecision],
    repeat_limit: int = 2,
    guidance: str | None = None,
    memory_limit: int = 12,
) -> dict[str, Any]:
    """Continue v2 from an exactly restored prefix without an extra environment step."""

    current_observation = str(observation)
    current_info = _normalized_info(info)
    current_history = list(history)
    current_score = float(score)
    task_id = _task_id(current_info)
    trace: list[dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0
    model_calls = 0
    invalid_actions = 0
    parser_failure = False
    success = bool(current_info.get("won", current_score > 0.0))
    termination_reason = "max_total_steps"

    for step_index in range(start_step, max_total_steps):
        commands = sorted(
            str(item) for item in current_info.get("admissible_commands", [])
        )
        if not commands:
            termination_reason = "no_admissible_commands"
            break
        policy_commands = anti_loop_commands(
            task=task,
            observation=current_observation,
            history=current_history,
            admissible_commands=commands,
            repeat_limit=repeat_limit,
        )
        prefix_state = _snapshot(
            current_observation,
            current_info,
            step_index,
            current_score,
        )
        messages = build_deliberation_messages(
            task=task,
            observation=current_observation,
            history=current_history,
            admissible_commands=policy_commands,
            guidance=guidance,
            memory_limit=memory_limit,
        )
        decision = decide(messages, policy_commands)
        input_tokens += int(decision.input_tokens)
        output_tokens += int(decision.output_tokens)
        model_calls += int(decision.model_calls)
        parsed = parse_admissible_action(decision.raw_output, policy_commands)
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
                "action_protocol": "source-policy-v2-command-trie",
                "prefix_state": prefix_state,
                "prefix_state_hash": state_hash(prefix_state),
                "policy_admissible_commands": policy_commands,
                "deliberation": decision.deliberation,
                "raw_output": decision.raw_output,
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
        "task_id": task_id,
        "success": success,
        "score": current_score,
        "steps": len(trace),
        "model_calls": model_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "invalid_actions": invalid_actions,
        "parser_failure": parser_failure,
        "termination_reason": termination_reason,
        "trace": trace,
    }
