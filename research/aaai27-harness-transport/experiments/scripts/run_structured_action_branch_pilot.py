#!/usr/bin/env python3
"""Run real Qwen/ALFWorld branches for structured action mechanisms."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from run_none_replan_branch_pilot import canonical_hash, git_state, restore_prefix
from run_source_policy_gate_v2 import load_decider
from traceh_core.baseline import _first, _normalized_info, _snapshot
from traceh_core.mechanisms import MechanismVariant, PrefixContext, propose_mechanism_action
from traceh_core.records import AppendOnlyRecordStore
from traceh_core.schema import load_schema, validate_instance
from traceh_core.source_policy_v2 import run_source_policy_continuation
from traceh_core.state import state_hash


ACTION_RECORD_NAMES = {
    MechanismVariant.NATURAL_REPLAN: "NATURAL_REPLAN",
    MechanismVariant.ANTI_LOOP_RETRY: "ANTI_LOOP_RETRY",
    MechanismVariant.PRECONDITION_CHECK: "PRECONDITION_CHECK",
    MechanismVariant.SUBGOAL_LEDGER: "SUBGOAL_LEDGER",
    MechanismVariant.BUNDLE_CONSERVATIVE: "BUNDLE_CONSERVATIVE",
}
ACTION_COSTS = {
    "NONE": 0.0,
    "NATURAL_REPLAN": 0.04,
    "ANTI_LOOP_RETRY": 0.02,
    "PRECONDITION_CHECK": 0.02,
    "SUBGOAL_LEDGER": 0.03,
    "BUNDLE_CONSERVATIVE": 0.04,
}


def select_candidates(
    candidate_report: dict[str, Any],
    *,
    baseline_offset: int | None,
    candidate_ids: set[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    candidates = list(candidate_report["candidates"])
    if baseline_offset is not None:
        marker = f"-o{baseline_offset:03d}-"
        candidates = [item for item in candidates if marker in item["baseline_run_id"]]
    if candidate_ids is not None:
        candidates = [item for item in candidates if item["candidate_id"] in candidate_ids]
    if limit is not None:
        candidates = candidates[:limit]
    if not candidates:
        raise ValueError("no selected candidates")
    return candidates


def run_none(
    candidate: dict[str, Any],
    trace_record: dict[str, Any],
    *,
    config_path: Path,
    data_dir: Path,
    env_seed: int,
    max_total_steps: int,
    decide: Any,
    repeat_limit: int,
) -> dict[str, Any]:
    env, task, observation, info, history, score, replayed_hash = restore_prefix(
        candidate,
        trace_record,
        config_path=config_path,
        data_dir=data_dir,
        env_seed=env_seed,
    )
    started = time.perf_counter()
    try:
        result = run_source_policy_continuation(
            env,
            task=task,
            observation=observation,
            info=info,
            history=history,
            score=score,
            start_step=int(candidate["step_index"]),
            max_total_steps=max_total_steps,
            decide=decide,
            repeat_limit=repeat_limit,
        )
    finally:
        env.close()
    result["wall_time_seconds"] = time.perf_counter() - started
    result["replayed_state_hash"] = replayed_hash
    result["mechanism"] = {"variant": "NONE", "intervene": False}
    result["forced_step"] = None
    return result


def run_structured(
    candidate: dict[str, Any],
    trace_record: dict[str, Any],
    *,
    variant: MechanismVariant,
    config_path: Path,
    data_dir: Path,
    env_seed: int,
    max_total_steps: int,
    decide: Any,
    repeat_limit: int,
) -> dict[str, Any]:
    env, task, observation, info, history, score, replayed_hash = restore_prefix(
        candidate,
        trace_record,
        config_path=config_path,
        data_dir=data_dir,
        env_seed=env_seed,
    )
    started = time.perf_counter()
    try:
        commands = tuple(sorted(str(item) for item in info.get("admissible_commands", [])))
        proposal = propose_mechanism_action(
            PrefixContext(
                task=task,
                observation=observation,
                history=tuple(history),
                admissible_commands=commands,
                event_type=str(candidate.get("event_type", "no_progress")),
            ),
            variant,
        )
        forced_step = None
        current_observation = observation
        current_info = dict(info)
        current_history = list(history)
        current_score = float(score)
        start_step = int(candidate["step_index"])
        if proposal.intervene and proposal.selected_command is not None:
            prefix_state = _snapshot(current_observation, current_info, start_step, current_score)
            next_observations, scores, dones, next_info_batch = env.step([proposal.selected_command])
            next_observation = str(_first(next_observations))
            current_score = float(_first(scores))
            done = bool(_first(dones))
            next_info = _normalized_info(next_info_batch)
            next_state = _snapshot(next_observation, next_info, start_step + 1, current_score)
            forced_step = {
                "step_index": start_step,
                "prefix_state": prefix_state,
                "prefix_state_hash": state_hash(prefix_state),
                "selected_action": proposal.selected_command,
                "next_observation": next_observation,
                "next_state_hash": state_hash(next_state),
                "score": current_score,
                "done": done,
                "mechanism_reasons": list(proposal.reasons),
                "mechanism_confidence": proposal.confidence,
            }
            current_history.append((proposal.selected_command, next_observation))
            current_observation = next_observation
            current_info = next_info
            start_step += 1
            if done:
                result = {
                    "task_id": trace_record["task_id"],
                    "success": bool(next_info.get("won", current_score > 0.0)),
                    "score": current_score,
                    "steps": 1,
                    "model_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "invalid_actions": 0,
                    "parser_failure": False,
                    "termination_reason": "environment_done",
                    "trace": [forced_step],
                }
            else:
                continuation = run_source_policy_continuation(
                    env,
                    task=task,
                    observation=current_observation,
                    info=current_info,
                    history=current_history,
                    score=current_score,
                    start_step=start_step,
                    max_total_steps=max_total_steps,
                    decide=decide,
                    repeat_limit=repeat_limit,
                )
                result = {
                    **continuation,
                    "steps": continuation["steps"] + 1,
                    "trace": [forced_step, *continuation["trace"]],
                }
        else:
            continuation = run_source_policy_continuation(
                env,
                task=task,
                observation=current_observation,
                info=current_info,
                history=current_history,
                score=current_score,
                start_step=start_step,
                max_total_steps=max_total_steps,
                decide=decide,
                repeat_limit=repeat_limit,
            )
            result = continuation
        result["wall_time_seconds"] = time.perf_counter() - started
        result["replayed_state_hash"] = replayed_hash
        result["mechanism"] = {
            "variant": variant.value,
            "record_action": ACTION_RECORD_NAMES[variant],
            "intervene": proposal.intervene,
            "selected_command": proposal.selected_command,
            "confidence": proposal.confidence,
            "reasons": list(proposal.reasons),
        }
        result["forced_step"] = forced_step
        return result
    finally:
        env.close()


def summarize_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actionable = [row for row in rows if row["action"] != "NONE"]
    positive = sum(row["raw_score_advantage"] > 1e-12 for row in actionable)
    negative = sum(row["raw_score_advantage"] < -1e-12 for row in actionable)
    return {
        "branch_count": len(rows),
        "actionable_branch_count": len(actionable),
        "positive_actionable_count": positive,
        "negative_actionable_count": negative,
        "all_actionable_raw_scores_zero": bool(actionable)
        and all(abs(row["action_score"]) <= 1e-12 for row in actionable),
        "best_actionable_advantage": max(
            (row["raw_score_advantage"] for row in actionable),
            default=0.0,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--baseline-offset", type=int)
    parser.add_argument("--candidate-id", action="append")
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--env-seed", type=int, default=20260712)
    parser.add_argument("--max-total-steps", type=int, default=50)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--deliberation-max-tokens", type=int, default=64)
    parser.add_argument("--repeat-limit", type=int, default=2)
    parser.add_argument(
        "--mechanism",
        action="append",
        choices=[variant.value for variant in ACTION_RECORD_NAMES],
        default=None,
    )
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    candidate_report = json.loads(args.candidates.read_text(encoding="utf-8"))
    candidates = select_candidates(
        candidate_report,
        baseline_offset=args.baseline_offset,
        candidate_ids=set(args.candidate_id) if args.candidate_id else None,
        limit=args.candidate_limit,
    )
    seeds = args.seed or [0]
    variants = [
        MechanismVariant(value)
        for value in (
            args.mechanism
            or [
                "natural_replan",
                "anti_loop_retry",
                "precondition_check",
                "subgoal_ledger",
                "bundle_conservative",
            ]
        )
    ]
    checkpoint_hash = hashlib.sha256(args.checkpoint_manifest.read_bytes()).hexdigest()
    harness_commit = git_state(args.project_root.resolve())
    frozen_config = {
        "run_label": args.run_label,
        "model_id": args.model_id,
        "checkpoint_hash": checkpoint_hash,
        "precision": "nf4-double-quant-bf16-compute",
        "source_policy": "SOURCE_POLICY_V2",
        "candidate_count": len(candidates),
        "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
        "branch_seeds": seeds,
        "mechanisms": [variant.value for variant in variants],
        "max_total_steps": args.max_total_steps,
        "max_input_tokens": args.max_input_tokens,
        "deliberation_max_tokens": args.deliberation_max_tokens,
        "repeat_limit": args.repeat_limit,
        "oracle_fields_allowed": [],
        "forced_action_steps": 1,
    }
    config_hash = canonical_hash(frozen_config)

    load_started = time.perf_counter()
    model, decide = load_decider(
        args.model.resolve(),
        max_input_tokens=args.max_input_tokens,
        deliberation_max_tokens=args.deliberation_max_tokens,
    )
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started

    trace_cache = {
        candidate["candidate_id"]: json.loads(Path(candidate["trace_path"]).read_text(encoding="utf-8"))
        for candidate in candidates
    }
    branch_schema = load_schema(args.project_root / "experiments/schemas/traceh-branch.schema.json")
    branch_store = AppendOnlyRecordStore(
        args.raw_root / "branches",
        validator=lambda record: validate_instance(record, branch_schema),
    )
    trace_store = AppendOnlyRecordStore(args.raw_root / "traces")
    existing_branches = {record["run_id"]: record for record in branch_store.iter_records()}
    existing_traces = {record["run_id"]: record for record in trace_store.iter_records()}

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        trace_record = trace_cache[candidate["candidate_id"]]
        for seed in seeds:
            none_run_id = f"{args.run_label}-{candidate['candidate_id']}-NONE-s{seed}"
            if args.reuse_existing and none_run_id in existing_branches and none_run_id in existing_traces:
                none_result = existing_traces[none_run_id]
                none_score = float(none_result["raw_terminal_score"])
                rows.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "seed": seed,
                        "action": "NONE",
                        "action_score": none_score,
                        "none_score": none_score,
                        "raw_score_advantage": 0.0,
                        "success": bool(none_result["success"]),
                        "mechanism": {"variant": "NONE", "intervene": False},
                    }
                )
            else:
                result = run_none(
                    candidate,
                    trace_record,
                    config_path=args.config,
                    data_dir=args.data,
                    env_seed=args.env_seed,
                    max_total_steps=args.max_total_steps,
                    decide=decide,
                    repeat_limit=args.repeat_limit,
                )
                none_score = float(result["score"])
                rows.append(
                    append_records(
                        branch_store,
                        trace_store,
                        run_id=none_run_id,
                        candidate=candidate,
                        model_id=args.model_id,
                        checkpoint_hash=checkpoint_hash,
                        config_hash=config_hash,
                        harness_commit=harness_commit,
                        action="NONE",
                        seed=seed,
                        result=result,
                        none_score=none_score,
                    )
                )
                print(json.dumps({"completed": rows[-1]}, ensure_ascii=False), flush=True)

            for variant in variants:
                action = ACTION_RECORD_NAMES[variant]
                run_id = f"{args.run_label}-{candidate['candidate_id']}-{action}-s{seed}"
                if args.reuse_existing and run_id in existing_branches and run_id in existing_traces:
                    trace_payload = existing_traces[run_id]
                    action_score = float(trace_payload["raw_terminal_score"])
                    row = {
                        "candidate_id": candidate["candidate_id"],
                        "seed": seed,
                        "action": action,
                        "action_score": action_score,
                        "none_score": none_score,
                        "raw_score_advantage": action_score - none_score,
                        "success": bool(trace_payload["success"]),
                        "mechanism": trace_payload.get("mechanism", {}),
                    }
                    rows.append(row)
                    continue
                result = run_structured(
                    candidate,
                    trace_record,
                    variant=variant,
                    config_path=args.config,
                    data_dir=args.data,
                    env_seed=args.env_seed,
                    max_total_steps=args.max_total_steps,
                    decide=decide,
                    repeat_limit=args.repeat_limit,
                )
                rows.append(
                    append_records(
                        branch_store,
                        trace_store,
                        run_id=run_id,
                        candidate=candidate,
                        model_id=args.model_id,
                        checkpoint_hash=checkpoint_hash,
                        config_hash=config_hash,
                        harness_commit=harness_commit,
                        action=action,
                        seed=seed,
                        result=result,
                        none_score=none_score,
                    )
                )
                print(json.dumps({"completed": rows[-1]}, ensure_ascii=False), flush=True)

    summary = summarize_pairs(rows)
    decision = "REVIEW_POSITIVE_STRUCTURED_ACTIONS" if summary["positive_actionable_count"] else "PIVOT_OR_STOP_STRUCTURED_ACTIONS"
    report = {
        "experiment_id": "L3-STRUCTURED-ACTION-BRANCH-PILOT",
        "frozen_config": frozen_config,
        "config_hash": config_hash,
        "harness_commit": harness_commit,
        "model_load_seconds": model_load_seconds,
        "model_memory_footprint_gib": model.get_memory_footprint() / 1024**3,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "ok": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def append_records(
    branch_store: AppendOnlyRecordStore,
    trace_store: AppendOnlyRecordStore,
    *,
    run_id: str,
    candidate: dict[str, Any],
    model_id: str,
    checkpoint_hash: str,
    config_hash: str,
    harness_commit: str,
    action: str,
    seed: int,
    result: dict[str, Any],
    none_score: float,
) -> dict[str, Any]:
    action_score = float(result["score"])
    action_cost = ACTION_COSTS[action]
    terminal_utility = action_score - action_cost
    branch_record = {
        "run_id": run_id,
        "baseline_run_id": candidate["baseline_run_id"],
        "task_id": candidate["task_id"],
        "model_id": model_id,
        "prefix_id": candidate["candidate_id"],
        "prefix_state_hash": candidate["prefix_state_hash"],
        "replayed_state_hash": result["replayed_state_hash"],
        "event_type": str(candidate.get("event_type", "no_progress")),
        "action": action,
        "seed": seed,
        "terminal_utility": terminal_utility,
        "none_utility": none_score,
        "action_advantage": terminal_utility - none_score,
        "steps": int(result["steps"]),
        "input_tokens": int(result["input_tokens"]),
        "output_tokens": int(result["output_tokens"]),
    }
    trace_payload = {
        "run_id": run_id,
        "config_hash": config_hash,
        "harness_commit": harness_commit,
        "checkpoint_hash": checkpoint_hash,
        "action_cost": action_cost,
        "raw_terminal_score": action_score,
        "success": bool(result["success"]),
        "mechanism": result["mechanism"],
        "forced_step": result["forced_step"],
        "parser_failure": bool(result["parser_failure"]),
        "invalid_actions": int(result["invalid_actions"]),
        "wall_time_seconds": float(result["wall_time_seconds"]),
        "termination_reason": str(result["termination_reason"]),
        "trace": result["trace"],
    }
    branch_store.append(branch_record)
    trace_store.append(trace_payload)
    return {
        "candidate_id": candidate["candidate_id"],
        "seed": seed,
        "action": action,
        "action_score": action_score,
        "none_score": none_score,
        "raw_score_advantage": action_score - none_score,
        "cost_adjusted_advantage": terminal_utility - none_score,
        "success": bool(result["success"]),
        "parser_failure": bool(result["parser_failure"]),
        "mechanism": result["mechanism"],
        "steps": int(result["steps"]),
        "input_tokens": int(result["input_tokens"]),
        "output_tokens": int(result["output_tokens"]),
    }


if __name__ == "__main__":
    main()
