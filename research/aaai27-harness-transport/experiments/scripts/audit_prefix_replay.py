#!/usr/bin/env python3
"""Replay selected baseline prefixes and verify exact TRACE-H state hashes."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml
from alfworld.agents.environment import get_environment

from traceh_core.state import diff_states, state_hash


OFFSET_RE = re.compile(r"-o(\d{3})-s\d+$")


def first(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def normalized_info(info: dict[str, Any]) -> dict[str, Any]:
    return {key: first(value) for key, value in info.items()}


def task_id(info: dict[str, Any]) -> str:
    for key in ("extra.gamefile", "gamefile", "extra.game_file"):
        if key in info:
            return str(info[key])
    return "unknown-game"


def snapshot(
    observation: str,
    info: dict[str, Any],
    step_index: int,
    score: float,
) -> dict[str, Any]:
    return {
        "task_id": task_id(info),
        "location": observation.splitlines()[0].strip() if observation else "",
        "inventory": [
            line.strip() for line in observation.splitlines() if "carrying" in line.casefold()
        ],
        "completed_predicates": [
            f"won={bool(info.get('won', False))}",
            f"score={score:.6f}",
        ],
        "admissible_commands": sorted(
            str(item) for item in info.get("admissible_commands", [])
        ),
        "step_index": step_index,
        "recent_tool_result": {
            "observation": observation,
            "won": bool(info.get("won", False)),
        },
    }


def build_env(config_path: Path, data_dir: Path, seed: int, offset: int):
    os.environ["ALFWORLD_DATA"] = str(data_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    env_class = get_environment(config["env"]["type"])
    base_env = env_class(config, train_eval="eval_in_distribution")
    env = base_env.init_env(batch_size=1)
    env.seed(seed)
    if offset:
        env.skip(offset)
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_report = json.loads(args.candidates.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidate_report["candidates"]:
        grouped.setdefault(candidate["baseline_run_id"], []).append(candidate)

    comparisons = []
    for baseline_run_id, candidates in sorted(grouped.items()):
        match = OFFSET_RE.search(baseline_run_id)
        if match is None:
            raise ValueError(f"cannot recover episode offset from {baseline_run_id}")
        offset = int(match.group(1))
        trace_path = Path(candidates[0]["trace_path"])
        trace_record = json.loads(trace_path.read_text(encoding="utf-8"))
        by_step = {int(candidate["step_index"]): candidate for candidate in candidates}

        env = build_env(args.config, args.data, args.seed, offset)
        observations, info_batch = env.reset()
        observation = str(first(observations))
        info = normalized_info(info_batch)
        score = 0.0
        try:
            for step_index in range(max(by_step) + 1):
                if step_index in by_step:
                    candidate = by_step[step_index]
                    replayed_state = snapshot(observation, info, step_index, score)
                    replayed_hash = state_hash(replayed_state)
                    original_state = trace_record["trace"][step_index]["prefix_state"]
                    comparisons.append(
                        {
                            "candidate_id": candidate["candidate_id"],
                            "baseline_run_id": baseline_run_id,
                            "step_index": step_index,
                            "expected_hash": candidate["prefix_state_hash"],
                            "replayed_hash": replayed_hash,
                            "match": replayed_hash == candidate["prefix_state_hash"],
                            "diff": diff_states(original_state, replayed_state),
                        }
                    )
                if step_index == max(by_step):
                    break
                action = trace_record["trace"][step_index]["selected_action"]
                observations, scores, dones, info_batch = env.step([action])
                observation = str(first(observations))
                score = float(first(scores))
                info = normalized_info(info_batch)
                if bool(first(dones)):
                    raise RuntimeError(
                        f"environment terminated before selected prefix {baseline_run_id}"
                    )
        finally:
            env.close()

    matching = sum(item["match"] for item in comparisons)
    report = {
        "experiment_id": "L3-PREFIX-REPLAY-AUDIT",
        "candidate_count": len(comparisons),
        "matching_prefixes": matching,
        "match_ratio": matching / len(comparisons) if comparisons else 0.0,
        "comparisons": comparisons,
        "ok": bool(comparisons) and matching == len(comparisons),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
