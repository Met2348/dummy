#!/usr/bin/env python3
"""Diagnostic visible-state symbolic search macro for ALFWorld prefixes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from run_none_replan_branch_pilot import restore_prefix
from traceh_core.baseline import _first, _normalized_info
from traceh_core.mechanisms import normalize_text, task_goal_text


CONTAINER_WORDS = (
    "drawer",
    "cabinet",
    "sidetable",
    "dresser",
    "sofa",
    "armchair",
    "garbagecan",
    "shelf",
    "countertop",
    "table",
)

SOURCE_FIRST_WORDS = (
    "dresser",
    "sofa",
    "sidetable",
    "armchair",
    "cabinet",
    "shelf",
    "countertop",
    "table",
    "garbagecan",
    "drawer",
)

KITCHEN_SOURCE_FIRST_WORDS = (
    "countertop",
    "fridge",
    "microwave",
    "diningtable",
    "cabinet",
    "sinkbasin",
    "garbagecan",
    "drawer",
)

ALL_SEARCH_WORDS = tuple(
    sorted({*CONTAINER_WORDS, *SOURCE_FIRST_WORDS, *KITCHEN_SOURCE_FIRST_WORDS})
)

TRANSFORM_APPLIANCES = {
    "clean": "sinkbasin",
    "cool": "fridge",
    "heat": "microwave",
}

FOOD_OBJECTS = {
    "apple",
    "bread",
    "egg",
    "lettuce",
    "potato",
    "tomato",
}


def parse_goal(task: str) -> dict[str, str | None]:
    goal = normalize_text(task_goal_text(task))
    ignored_objects = {"a", "an", "it", "one", "some", "them", "two"}
    object_value = None
    for pattern in (
        r"\b(?:put|place)\s+(?:a|an|some|two|one)?\s*([a-z][a-z0-9_]*)\b",
        r"\b(?:heat|cool|clean)\s+(?:a|an|some|two|one)?\s*([a-z][a-z0-9_]*)\b",
        r"\b(?:find|get|take)\s+(?:a|an|some|two|one)?\s*([a-z][a-z0-9_]*)\b",
    ):
        match = re.search(pattern, goal)
        if match and match.group(1) not in ignored_objects:
            object_value = match.group(1)
            break
    target_match = re.search(r"\b(?:in|on)\s+([a-z][a-z0-9_]*)\b", goal)
    transform_match = re.search(r"\b(heat|cool|clean)\b", goal)
    count = 2 if "two" in goal else 1
    return {
        "goal": goal,
        "object": object_value,
        "target_receptacle": target_match.group(1) if target_match else None,
        "transform": transform_match.group(1) if transform_match else None,
        "count": str(count),
    }


def action_place(normalized_action: str) -> str | None:
    if normalized_action.startswith("go to "):
        return normalized_action.removeprefix("go to ").strip()
    if normalized_action.startswith("open "):
        return normalized_action.removeprefix("open ").strip()
    if normalized_action.startswith("close "):
        return normalized_action.removeprefix("close ").strip()
    return None


def visible_places(text: str) -> set[str]:
    normalized = normalize_text(text)
    places: set[str] = set()
    for pattern in (
        r"\barrive at ([a-z][a-z0-9_]*\s+\d+)\b",
        r"\bon the ([a-z][a-z0-9_]*\s+\d+)\b",
        r"\bthe ([a-z][a-z0-9_]*\s+\d+) is (?:open|closed)\b",
        r"\bin the ([a-z][a-z0-9_]*\s+\d+)\b",
    ):
        places.update(re.findall(pattern, normalized))
    return places


def object_instance_from_action(normalized_action: str, target_object: str | None = None) -> str | None:
    if target_object:
        match = re.search(rf"\b{re.escape(target_object)}\s+\d+\b", normalized_action)
        if match:
            return match.group(0)
        return None
    match = re.search(r"\b(?:clean|cool|heat|take|move|put)\s+([a-z][a-z0-9_]*\s+\d+)\b", normalized_action)
    return match.group(1) if match else None


def delivery_target_from_action(normalized_action: str) -> str | None:
    match = re.search(r"\b(?:to|in|on)\s+([a-z][a-z0-9_]*\s+\d+)\b", normalized_action)
    return match.group(1) if match else None


def infer_carried_object(commands: list[str], target_object: str | None) -> str | None:
    if not target_object:
        return None
    for command in commands:
        normalized = normalize_text(command)
        if normalized.startswith(("move ", "put ", "clean ", "cool ", "heat ")):
            instance = object_instance_from_action(normalized, target_object)
            if instance:
                return instance
    return None


def object_visible_in_text(text: str, target_object: str | None) -> bool:
    if not target_object:
        return False
    normalized = normalize_text(text)
    return bool(re.search(rf"\b{re.escape(target_object)}\s+\d+\b", normalized))


def target_actionable_commands(commands: list[str], target_object: str | None) -> list[str]:
    if not target_object:
        return []
    actionable: list[str] = []
    for command in commands:
        normalized = normalize_text(command)
        if object_instance_from_action(normalized, target_object):
            actionable.append(command)
    return actionable


def initial_ledgers(
    history: list[tuple[str, str]] | tuple[tuple[str, str], ...],
    target_object: str | None,
    target_receptacle: str | None,
) -> tuple[set[str], set[str], str | None]:
    delivered_objects: set[str] = set()
    transformed_objects: set[str] = set()
    deposit_target: str | None = None
    for action, _observation in history:
        normalized = normalize_text(action)
        if normalized.startswith(("clean ", "cool ", "heat ")):
            transformed = object_instance_from_action(normalized, target_object)
            if transformed:
                transformed_objects.add(transformed)
        if normalized.startswith(("move ", "put ")):
            delivered = object_instance_from_action(normalized, target_object)
            target = delivery_target_from_action(normalized)
            if delivered and target_receptacle and target and target_receptacle in target:
                delivered_objects.add(delivered)
                deposit_target = deposit_target or target
        if normalized.startswith("take "):
            taken = object_instance_from_action(normalized, target_object)
            if taken and taken in delivered_objects:
                delivered_objects.remove(taken)
    return delivered_objects, transformed_objects, deposit_target


def target_place_commands(commands: list[tuple[str, str]], target_receptacle: str) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for command, normalized in commands:
        place = action_place(normalized)
        if place and target_receptacle in place:
            rows.append((command, normalized, place))
    return rows


def ordered_search_words(
    *,
    target_object: str | None,
    target_receptacle: str | None,
    transform: str | None,
    search_order: str,
    global_step_index: int,
    horizon_switch_step: int,
) -> list[str]:
    effective_order = search_order
    if search_order == "source_horizon_aware" and global_step_index >= horizon_switch_step:
        effective_order = "reverse_container"
    elif search_order == "source_horizon_aware":
        effective_order = "source_first"

    if effective_order == "source_first" and (transform or target_object in FOOD_OBJECTS):
        raw_words = (*KITCHEN_SOURCE_FIRST_WORDS, target_receptacle)
    elif effective_order == "source_first":
        raw_words = (*SOURCE_FIRST_WORDS, target_receptacle)
    elif effective_order == "container_first":
        raw_words = (*CONTAINER_WORDS, target_receptacle)
    elif effective_order == "reverse_container":
        raw_words = (*reversed(CONTAINER_WORDS), target_receptacle)
    elif effective_order == "kitchen_first":
        raw_words = (*KITCHEN_SOURCE_FIRST_WORDS, target_receptacle)
    elif effective_order == "furniture_first":
        raw_words = (*SOURCE_FIRST_WORDS, target_receptacle)
    elif effective_order == "alphabetical":
        raw_words = (*ALL_SEARCH_WORDS, target_receptacle)
    else:
        raw_words = (target_receptacle, *CONTAINER_WORDS)
    words: list[str] = []
    for word in raw_words:
        if word and word not in words:
            words.append(word)
    return words


def choose_action(
    *,
    task: str,
    observation: str,
    commands: list[str],
    visited: set[str],
    inventory: list[str],
    delivered_objects: set[str],
    transformed_objects: set[str],
    deposit_target: str | None,
    search_order: str,
    global_step_index: int,
    horizon_switch_step: int,
) -> tuple[str | None, str]:
    goal = parse_goal(task)
    target_object = goal["object"]
    target_receptacle = goal["target_receptacle"]
    transform = goal["transform"]
    normalized_observation = normalize_text(observation)
    normalized_commands = [(command, normalize_text(command)) for command in commands]
    current_places = visible_places(observation)

    if not inventory and target_object and target_receptacle and not transform:
        for command, normalized in normalized_commands:
            if (
                normalized.startswith(("move ", "put "))
                and target_object in normalized
                and target_receptacle in normalized
            ):
                return command, "deliver_inferred_inventory_to_target_receptacle"

    if target_object:
        for command, normalized in normalized_commands:
            instance = object_instance_from_action(normalized, target_object)
            if (
                normalized.startswith(f"take {target_object} ")
                and (instance is None or instance not in delivered_objects)
            ):
                return command, "take_visible_target_object"

    if not inventory and target_object and transform and transformed_objects:
        appliance = TRANSFORM_APPLIANCES.get(transform)
        if appliance:
            for command, normalized in normalized_commands:
                place = action_place(normalized)
                if normalized.startswith("open ") and place and appliance in place and place in current_places:
                    return command, f"open_{appliance}_to_recover_transformed_object"
            for command, normalized in normalized_commands:
                place = action_place(normalized)
                if normalized.startswith("go to ") and place and appliance in place:
                    return command, f"return_to_{appliance}_to_recover_transformed_object"

    if inventory and target_object and transform:
        carried_instance = object_instance_from_action(normalize_text(inventory[-1]), target_object)
        if carried_instance is None or carried_instance not in transformed_objects:
            for command, normalized in normalized_commands:
                if normalized.startswith(f"{transform} ") and target_object in normalized:
                    return command, f"{transform}_carried_target_object"
            appliance = TRANSFORM_APPLIANCES.get(transform)
            if appliance:
                for command, normalized in normalized_commands:
                    place = action_place(normalized)
                    if normalized.startswith("open ") and place and appliance in place and place in current_places:
                        return command, f"open_{appliance}_for_{transform}"
                for command, normalized in normalized_commands:
                    place = action_place(normalized)
                    if normalized.startswith("go to ") and place and appliance in place:
                        return command, f"go_to_{appliance}_for_{transform}"

    if inventory and target_receptacle:
        if deposit_target:
            for command, normalized in normalized_commands:
                if normalized.startswith(("move ", "put ")) and deposit_target in normalized:
                    return command, "deliver_carried_object_to_locked_deposit_target"
            for command, normalized in normalized_commands:
                place = action_place(normalized)
                if normalized.startswith("open ") and place == deposit_target and place in current_places:
                    return command, "open_locked_deposit_target_with_inventory"
            for command, normalized in normalized_commands:
                place = action_place(normalized)
                if normalized.startswith("go to ") and place == deposit_target:
                    return command, "return_to_locked_deposit_target_with_inventory"
        for command, normalized in normalized_commands:
            if normalized.startswith(("move ", "put ")) and target_receptacle in normalized:
                return command, "deliver_carried_object_to_target_receptacle"
        for command, normalized, place in target_place_commands(normalized_commands, target_receptacle):
            if normalized.startswith("open ") and place in current_places:
                return command, "open_current_target_receptacle_with_inventory"
        for command, normalized, place in target_place_commands(normalized_commands, target_receptacle):
            if normalized.startswith("go to ") and place not in visited:
                return command, "go_to_unvisited_target_receptacle_with_inventory"
        for command, normalized, _place in target_place_commands(normalized_commands, target_receptacle):
            if normalized.startswith("go to "):
                return command, "return_to_target_receptacle_with_inventory"

    for command, normalized in normalized_commands:
        place = action_place(normalized)
        if normalized.startswith("open ") and "closed" in normalized_observation:
            if place is None or place in current_places:
                return command, "open_closed_current_container"

    prioritized_words = ordered_search_words(
        target_object=target_object,
        target_receptacle=target_receptacle,
        transform=transform,
        search_order=search_order,
        global_step_index=global_step_index,
        horizon_switch_step=horizon_switch_step,
    )
    for word in prioritized_words:
        for command, normalized in normalized_commands:
            if not normalized.startswith("go to "):
                continue
            place = action_place(normalized) or ""
            if word in place and place not in visited:
                return command, f"go_to_unvisited_{word}"

    if search_order == "source_exhaustive":
        for command, normalized in normalized_commands:
            if not normalized.startswith("go to "):
                continue
            place = action_place(normalized)
            if place and place not in visited:
                return command, "go_to_any_unvisited_place_after_source_priority"

    for command, normalized in normalized_commands:
        if normalized.startswith("look"):
            return command, "fallback_look"
    return None, "no_supported_action"


def run_macro(
    candidate: dict[str, Any],
    trace_record: dict[str, Any],
    *,
    config_path: Path,
    data_dir: Path,
    env_seed: int,
    max_macro_steps: int,
    search_order: str,
    policy_variant: str,
    horizon_switch_step: int,
) -> dict[str, Any]:
    env, task, observation, info, history, score, replayed_hash = restore_prefix(
        candidate,
        trace_record,
        config_path=config_path,
        data_dir=data_dir,
        env_seed=env_seed,
    )
    goal = parse_goal(task)
    visited: set[str] = set()
    inventory: list[str] = []
    delivered_objects, transformed_objects, deposit_target = initial_ledgers(
        history,
        goal["object"],
        goal["target_receptacle"],
    )
    if policy_variant == "no_history_ledger":
        delivered_objects = set()
        transformed_objects = set()
        deposit_target = None
    trace = []
    success = bool(info.get("won", score > 0.0))
    try:
        for local_step in range(max_macro_steps):
            global_step_index = int(candidate["step_index"]) + local_step
            commands = sorted(str(command) for command in info.get("admissible_commands", []))
            target_visible_before = object_visible_in_text(observation, goal["object"])
            target_actionable_before = target_actionable_commands(commands, goal["object"])
            inventory_before = list(inventory)
            delivered_before = sorted(delivered_objects)
            transformed_before = sorted(transformed_objects)
            deposit_target_before = deposit_target
            if policy_variant != "no_inventory_inference":
                inferred_inventory = infer_carried_object(commands, goal["object"])
                if inferred_inventory and not inventory and inferred_inventory not in delivered_objects:
                    inventory.append(inferred_inventory)
            choice_delivered = set() if policy_variant == "no_instance_ledger" else delivered_objects
            choice_transformed = set() if policy_variant == "no_instance_ledger" else transformed_objects
            choice_deposit_target = None if policy_variant == "no_deposit_lock" else deposit_target
            action, reason = choose_action(
                task=task,
                observation=observation,
                commands=commands,
                visited=visited,
                inventory=inventory,
                delivered_objects=choice_delivered,
                transformed_objects=choice_transformed,
                deposit_target=choice_deposit_target,
                search_order=search_order,
                global_step_index=global_step_index,
                horizon_switch_step=horizon_switch_step,
            )
            if action is None:
                break
            normalized = normalize_text(action)
            place = action_place(normalized)
            if place:
                visited.add(place)
            observations, scores, dones, info_batch = env.step([action])
            observation = str(_first(observations))
            score = float(_first(scores))
            done = bool(_first(dones))
            info = _normalized_info(info_batch)
            success = bool(info.get("won", score > 0.0))
            if normalized.startswith("take "):
                inventory.append(action)
            if normalized.startswith(("clean ", "cool ", "heat ")):
                transformed = object_instance_from_action(normalized, goal["object"])
                if transformed and policy_variant != "no_instance_ledger":
                    transformed_objects.add(transformed)
            if normalized.startswith(("move ", "put ")):
                delivered = object_instance_from_action(normalized, goal["object"])
                if delivered and policy_variant != "no_instance_ledger":
                    delivered_objects.add(delivered)
                if policy_variant != "no_deposit_lock":
                    deposit_target = deposit_target or delivery_target_from_action(normalized)
                if inventory:
                    inventory.pop()
            trace.append(
                {
                    "global_step_index": int(candidate["step_index"]) + local_step,
                    "action": action,
                    "reason": reason,
                    "horizon_switch_active": search_order == "source_horizon_aware"
                    and global_step_index >= horizon_switch_step,
                    "score": score,
                    "done": done,
                    "success": success,
                    "target_visible_before": target_visible_before,
                    "target_actionable_commands_before": target_actionable_before,
                    "inventory_before": inventory_before,
                    "delivered_before": delivered_before,
                    "transformed_before": transformed_before,
                    "deposit_target_before": deposit_target_before,
                    "observation": observation,
                }
            )
            if done:
                break
    finally:
        env.close()
    return {
        "candidate_id": candidate["candidate_id"],
        "baseline_run_id": candidate["baseline_run_id"],
        "task_id": candidate["task_id"],
        "prefix_state_hash": candidate["prefix_state_hash"],
        "replayed_state_hash": replayed_hash,
        "goal": goal,
        "max_macro_steps": max_macro_steps,
        "search_order": search_order,
        "policy_variant": policy_variant,
        "horizon_switch_step": horizon_switch_step,
        "macro_steps": len(trace),
        "terminal_score": score,
        "success": success,
        "visited_count": len(visited),
        "delivered_objects": sorted(delivered_objects),
        "transformed_objects": sorted(transformed_objects),
        "deposit_target": deposit_target,
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-offset", type=int)
    parser.add_argument("--candidate-id", action="append")
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--env-seed", type=int, default=20260712)
    parser.add_argument("--max-macro-steps", type=int, default=30)
    parser.add_argument(
        "--search-order",
        choices=(
            "target_first",
            "source_first",
            "source_exhaustive",
            "source_horizon_aware",
            "container_first",
            "reverse_container",
            "kitchen_first",
            "furniture_first",
            "alphabetical",
        ),
        default="target_first",
    )
    parser.add_argument("--horizon-switch-step", type=int, default=40)
    parser.add_argument(
        "--policy-variant",
        choices=(
            "full",
            "no_history_ledger",
            "no_deposit_lock",
            "no_inventory_inference",
            "no_instance_ledger",
        ),
        default="full",
    )
    args = parser.parse_args()

    report = json.loads(args.candidates.read_text(encoding="utf-8"))
    candidates = list(report["candidates"])
    if args.baseline_offset is not None:
        marker = f"-o{args.baseline_offset:03d}-"
        candidates = [item for item in candidates if marker in item["baseline_run_id"]]
    if args.candidate_id:
        keep = set(args.candidate_id)
        candidates = [item for item in candidates if item["candidate_id"] in keep]
    if args.candidate_limit is not None:
        candidates = candidates[: args.candidate_limit]
    if not candidates:
        raise ValueError("no selected candidates")

    rows = []
    for candidate in candidates:
        trace_record = json.loads(Path(candidate["trace_path"]).read_text(encoding="utf-8"))
        row = run_macro(
            candidate,
            trace_record,
            config_path=args.config,
            data_dir=args.data,
            env_seed=args.env_seed,
            max_macro_steps=args.max_macro_steps,
            search_order=args.search_order,
            policy_variant=args.policy_variant,
            horizon_switch_step=args.horizon_switch_step,
        )
        rows.append(row)
        print(json.dumps({"completed": {k: row[k] for k in ("candidate_id", "macro_steps", "terminal_score", "success")}}, ensure_ascii=False), flush=True)

    result = {
        "experiment_id": "L3-SYMBOLIC-SEARCH-MACRO-PROBE",
        "max_macro_steps": args.max_macro_steps,
        "search_order": args.search_order,
        "policy_variant": args.policy_variant,
        "horizon_switch_step": args.horizon_switch_step,
        "candidate_count": len(rows),
        "success_count": sum(row["success"] for row in rows),
        "positive_score_count": sum(float(row["terminal_score"]) > 0.0 for row in rows),
        "rows": rows,
        "ok": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "candidate_count": result["candidate_count"],
                "success_count": result["success_count"],
                "positive_score_count": result["positive_score_count"],
                "search_order": result["search_order"],
                "policy_variant": result["policy_variant"],
                "horizon_switch_step": result["horizon_switch_step"],
                "max_macro_steps": result["max_macro_steps"],
                "ok": result["ok"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
