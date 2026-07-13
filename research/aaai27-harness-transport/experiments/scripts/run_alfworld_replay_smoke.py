#!/usr/bin/env python3
"""Run a no-LLM ALFWorld reset/step/replay state-hash smoke test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml
from alfworld.agents.environment import get_environment

from traceh_core.state import diff_states, state_hash


def first(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def normalized_info(info: dict[str, Any]) -> dict[str, Any]:
    return {key: first(value) for key, value in info.items()}


def task_id_from_info(info: dict[str, Any]) -> str:
    for key in ("extra.gamefile", "gamefile", "extra.game_file"):
        if key in info:
            return str(info[key])
    return "unknown-game"


def snapshot(
    observation: str,
    info: dict[str, Any],
    step_index: int,
    score: float = 0.0,
) -> dict[str, Any]:
    commands = sorted(str(item) for item in info.get("admissible_commands", []))
    inventory_markers = [line.strip() for line in observation.splitlines() if "carrying" in line.lower()]
    return {
        "task_id": task_id_from_info(info),
        "location": observation.splitlines()[0].strip() if observation else "",
        "inventory": inventory_markers,
        "completed_predicates": [f"won={bool(info.get('won', False))}", f"score={float(score):.6f}"],
        "admissible_commands": commands,
        "step_index": step_index,
        "recent_tool_result": {"observation": observation, "won": bool(info.get("won", False))},
    }


def build_env(config_path: Path, data_dir: Path, seed: int, skip_n: int):
    os.environ["ALFWORLD_DATA"] = str(data_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    env_class = get_environment(config["env"]["type"])
    base_env = env_class(config, train_eval="eval_in_distribution")
    env = base_env.init_env(batch_size=1)
    env.seed(seed)
    if skip_n:
        env.skip(skip_n)
    return base_env, env


def run_trajectory(
    config: Path,
    data: Path,
    seed: int,
    skip_n: int,
    actions: list[str] | None,
    steps: int,
):
    base_env, env = build_env(config, data, seed, skip_n)
    observations, info_batch = env.reset()
    info = normalized_info(info_batch)
    observation = str(first(observations))
    states = [snapshot(observation, info, 0)]
    chosen: list[str] = []
    try:
        for index in range(steps):
            candidates = sorted(str(item) for item in info.get("admissible_commands", []))
            if not candidates:
                break
            if actions is None:
                nontrivial = [item for item in candidates if item not in {"look", "inventory"}]
                pool = nontrivial or candidates
                action = pool[index % len(pool)]
            else:
                action = actions[index]
                if action not in candidates:
                    raise RuntimeError(f"recorded action is no longer admissible: {action}")
            observations, scores, dones, info_batch = env.step([action])
            info = normalized_info(info_batch)
            observation = str(first(observations))
            score = float(first(scores))
            states.append(snapshot(observation, info, index + 1, score))
            chosen.append(action)
            if bool(first(dones)):
                break
    finally:
        env.close()
    return {
        "num_games": int(base_env.num_games),
        "actions": chosen,
        "states": states,
        "hashes": [state_hash(state) for state in states],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    episodes = []
    all_comparisons = []
    for offset in range(args.episodes):
        original = run_trajectory(args.config, args.data, args.seed, offset, None, args.steps)
        replay = run_trajectory(
            args.config,
            args.data,
            args.seed,
            offset,
            original["actions"],
            len(original["actions"]),
        )
        comparisons = []
        for index, (left, right) in enumerate(zip(original["states"], replay["states"], strict=True)):
            comparison = {
                "episode_offset": offset,
                "step_index": index,
                "match": original["hashes"][index] == replay["hashes"][index],
                "original_hash": original["hashes"][index],
                "replay_hash": replay["hashes"][index],
                "diff": diff_states(left, right),
            }
            comparisons.append(comparison)
            all_comparisons.append(comparison)
        episodes.append(
            {
                "episode_offset": offset,
                "task_id": original["states"][0]["task_id"],
                "actions": original["actions"],
                "comparisons": comparisons,
                "ok": all(item["match"] for item in comparisons),
            }
        )
    if not all_comparisons:
        raise RuntimeError("replay smoke produced no state comparisons")
    ratio = sum(item["match"] for item in all_comparisons) / len(all_comparisons)
    task_ids = [item["task_id"] for item in episodes]
    unique_task_count = len(set(task_ids))
    report = {
        "experiment_id": "ALFWORLD-NO-LLM-REPLAY-SMOKE",
        "seed": args.seed,
        "requested_steps": args.steps,
        "episodes": args.episodes,
        "unique_task_count": unique_task_count,
        "state_checks": len(all_comparisons),
        "matching_state_checks": sum(item["match"] for item in all_comparisons),
        "replay_hash_ratio": ratio,
        "episode_results": episodes,
        "ok": (
            ratio >= 0.95
            and all(item["ok"] for item in episodes)
            and unique_task_count == args.episodes
            and "unknown-game" not in task_ids
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
