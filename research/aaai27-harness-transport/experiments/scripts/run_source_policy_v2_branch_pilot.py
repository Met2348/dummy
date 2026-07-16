#!/usr/bin/env python3
"""Run paired NONE/REPLAN branches on Source Policy v2 failure prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from run_none_replan_branch_pilot import (
    build_replan_messages,
    canonical_hash,
    generate_plan,
    git_state,
    restore_prefix,
)
from run_source_policy_gate_v2 import load_decider
from traceh_core.branch_analysis import decide_branch_pilot
from traceh_core.records import AppendOnlyRecordStore
from traceh_core.schema import load_schema, validate_instance
from traceh_core.source_policy_v2 import run_source_policy_continuation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--baseline-offset", type=int, required=True)
    parser.add_argument("--candidate-id", action="append")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--integrity-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--env-seed", type=int, default=20260712)
    parser.add_argument("--max-total-steps", type=int, default=50)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--deliberation-max-tokens", type=int, default=64)
    parser.add_argument("--replan-max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--repeat-limit", type=int, default=2)
    parser.add_argument("--replan-cost", type=float, default=0.03)
    parser.add_argument("--reuse-integrity", action="store_true")
    args = parser.parse_args()

    candidate_report = json.loads(args.candidates.read_text(encoding="utf-8"))
    offset_marker = f"-o{args.baseline_offset:03d}-"
    candidates = [
        candidate
        for candidate in candidate_report["candidates"]
        if offset_marker in candidate["baseline_run_id"]
    ]
    if args.candidate_id:
        candidate_ids = set(args.candidate_id)
        candidates = [
            candidate
            for candidate in candidates
            if candidate["candidate_id"] in candidate_ids
        ]
    if args.candidate_id and not candidates:
        raise ValueError("no selected candidates")
    if not args.candidate_id and len(candidates) != 2:
        raise ValueError(f"expected two failure-prefix candidates, found {len(candidates)}")

    checkpoint_hash = hashlib.sha256(args.checkpoint_manifest.read_bytes()).hexdigest()
    harness_commit = git_state(args.project_root.resolve())
    frozen_config = {
        "run_label": args.run_label,
        "model_id": args.model_id,
        "checkpoint_hash": checkpoint_hash,
        "source_policy": "SOURCE_POLICY_V2",
        "precision": "nf4-double-quant-bf16-compute",
        "baseline_offset": args.baseline_offset,
        "candidate_count": len(candidates),
        "branch_seeds": [0, 1, 2],
        "max_total_steps": args.max_total_steps,
        "max_input_tokens": args.max_input_tokens,
        "deliberation_max_tokens": args.deliberation_max_tokens,
        "repeat_limit": args.repeat_limit,
        "replan_max_tokens": args.replan_max_tokens,
        "replan_temperature": args.temperature,
        "replan_extra_calls": 1,
        "replan_cost": args.replan_cost,
        "executor_sampling": "deterministic",
        "none_execution_per_candidate": 1,
        "none_reused_across_seeds": True,
    }
    config_hash = canonical_hash(frozen_config)
    load_started = time.perf_counter()
    model, decide = load_decider(
        args.model.resolve(),
        max_input_tokens=args.max_input_tokens,
        deliberation_max_tokens=args.deliberation_max_tokens,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model.resolve(), local_files_only=True)
    tokenizer.truncation_side = "left"
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started

    trace_cache = {
        candidate["candidate_id"]: json.loads(
            Path(candidate["trace_path"]).read_text(encoding="utf-8")
        )
        for candidate in candidates
    }

    expected_candidate_ids = {candidate["candidate_id"] for candidate in candidates}
    if args.reuse_integrity:
        integrity_report = json.loads(args.integrity_report.read_text(encoding="utf-8"))
        observed_candidate_ids = {row["candidate_id"] for row in integrity_report["rows"]}
        if not integrity_report["ok"] or observed_candidate_ids != expected_candidate_ids:
            raise ValueError("existing integrity report does not match selected candidates")
    else:
        integrity_rows = []
        for candidate in candidates:
            trace_record = trace_cache[candidate["candidate_id"]]
            env, task, observation, info, history, score, replayed_hash = restore_prefix(
                candidate,
                trace_record,
                config_path=args.config,
                data_dir=args.data,
                env_seed=args.env_seed,
            )
            try:
                result = run_source_policy_continuation(
                    env,
                    task=task,
                    observation=observation,
                    info=info,
                    history=history,
                    score=score,
                    start_step=int(candidate["step_index"]),
                    max_total_steps=args.max_total_steps,
                    decide=decide,
                    repeat_limit=args.repeat_limit,
                )
            finally:
                env.close()
            original_suffix = [
                item["selected_action"]
                for item in trace_record["trace"][int(candidate["step_index"]) :]
            ]
            replayed_suffix = [item["selected_action"] for item in result["trace"]]
            row = {
                "candidate_id": candidate["candidate_id"],
                "prefix_hash_match": replayed_hash == candidate["prefix_state_hash"],
                "suffix_action_match": replayed_suffix == original_suffix,
                "terminal_score_match": result["score"]
                == float(trace_record["trace"][-1]["score"]),
                "expected_suffix_steps": len(original_suffix),
                "replayed_suffix_steps": len(replayed_suffix),
            }
            row["ok"] = all(
                row[key]
                for key in ("prefix_hash_match", "suffix_action_match", "terminal_score_match")
            )
            integrity_rows.append(row)
            print(json.dumps({"integrity": row}, ensure_ascii=False), flush=True)

        integrity_report = {
            "experiment_id": "L3-SOURCE-POLICY-V2-NONE-INTEGRITY",
            "candidate_count": len(integrity_rows),
            "matching_continuations": sum(row["ok"] for row in integrity_rows),
            "rows": integrity_rows,
            "ok": all(row["ok"] for row in integrity_rows),
        }
        args.integrity_report.parent.mkdir(parents=True, exist_ok=True)
        args.integrity_report.write_text(
            json.dumps(integrity_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if not integrity_report["ok"]:
        raise SystemExit(1)

    branch_schema = load_schema(args.project_root / "experiments/schemas/traceh-branch.schema.json")
    branch_store = AppendOnlyRecordStore(
        args.raw_root / "branches",
        validator=lambda record: validate_instance(record, branch_schema),
    )
    trace_store = AppendOnlyRecordStore(args.raw_root / "traces")
    pair_rows = []
    existing_branches = {
        record["run_id"]: record for record in branch_store.iter_records()
    }
    existing_traces = {record["run_id"]: record for record in trace_store.iter_records()}

    for candidate in candidates:
        trace_record = trace_cache[candidate["candidate_id"]]
        existing_none_ids = [
            run_id
            for run_id in existing_traces
            if f"-{candidate['candidate_id']}-none-s" in run_id
            and run_id in existing_branches
        ]
        if existing_none_ids:
            existing_id = sorted(existing_none_ids)[0]
            existing_trace = existing_traces[existing_id]
            existing_branch = existing_branches[existing_id]
            none_result = {
                "score": float(existing_trace["raw_terminal_score"]),
                "success": bool(existing_trace["success"]),
                "steps": int(existing_branch["steps"]),
                "model_calls": int(existing_trace["total_model_calls"]),
                "input_tokens": int(existing_branch["input_tokens"]),
                "output_tokens": int(existing_branch["output_tokens"]),
                "invalid_actions": int(existing_trace["invalid_actions"]),
                "parser_failure": bool(existing_trace["parser_failure"]),
                "termination_reason": existing_trace["termination_reason"],
                "trace": existing_trace["trace"],
                "wall_time_seconds": float(existing_trace["wall_time_seconds"]),
                "replayed_state_hash": existing_branch["replayed_state_hash"],
                "plan": None,
                "extra_model_calls": 0,
            }
        else:
            env, task, observation, info, history, score, replayed_hash = restore_prefix(
                candidate,
                trace_record,
                config_path=args.config,
                data_dir=args.data,
                env_seed=args.env_seed,
            )
            none_started = time.perf_counter()
            try:
                none_result = run_source_policy_continuation(
                    env,
                    task=task,
                    observation=observation,
                    info=info,
                    history=history,
                    score=score,
                    start_step=int(candidate["step_index"]),
                    max_total_steps=args.max_total_steps,
                    decide=decide,
                    repeat_limit=args.repeat_limit,
                )
            finally:
                env.close()
            none_result["wall_time_seconds"] = time.perf_counter() - none_started
            none_result["replayed_state_hash"] = replayed_hash
            none_result["plan"] = None
            none_result["extra_model_calls"] = 0
        for branch_seed in (0, 1, 2):
            none_run_id = (
                f"{args.run_label}-{candidate['candidate_id']}-none-s{branch_seed}"
            )
            replan_run_id = (
                f"{args.run_label}-{candidate['candidate_id']}-replan-s{branch_seed}"
            )
            completed = {
                run_id
                for run_id in (none_run_id, replan_run_id)
                if run_id in existing_branches and run_id in existing_traces
            }
            if completed:
                if completed != {none_run_id, replan_run_id}:
                    raise RuntimeError("incomplete existing branch pair cannot be resumed")
                none_trace = existing_traces[none_run_id]
                replan_trace = existing_traces[replan_run_id]
                none_score = float(none_trace["raw_terminal_score"])
                replan_score = float(replan_trace["raw_terminal_score"])
                pair_rows.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "seed": branch_seed,
                        "none_score": none_score,
                        "replan_score": replan_score,
                        "raw_score_advantage": replan_score - none_score,
                        "none_success": bool(none_trace["success"]),
                        "replan_success": bool(replan_trace["success"]),
                    }
                )
                print(json.dumps({"resumed": pair_rows[-1]}, ensure_ascii=False), flush=True)
                continue
            action_results: dict[str, dict[str, Any]] = {"NONE": none_result}
            for action in ("REPLAN",):
                env, task, observation, info, history, score, replayed_hash = restore_prefix(
                    candidate,
                    trace_record,
                    config_path=args.config,
                    data_dir=args.data,
                    env_seed=args.env_seed,
                )
                plan_result = None
                guidance = None
                if action == "REPLAN":
                    plan_result = generate_plan(
                        tokenizer,
                        model,
                        build_replan_messages(
                            task=task,
                            observation=observation,
                            history=history,
                            admissible_commands=sorted(
                                str(item) for item in info.get("admissible_commands", [])
                            ),
                        ),
                        seed=branch_seed,
                        max_input_tokens=args.max_input_tokens,
                        max_new_tokens=args.replan_max_tokens,
                        temperature=args.temperature,
                    )
                    guidance = plan_result.raw_output
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
                        max_total_steps=args.max_total_steps,
                        decide=decide,
                        repeat_limit=args.repeat_limit,
                        guidance=guidance,
                    )
                finally:
                    env.close()
                result["wall_time_seconds"] = time.perf_counter() - started
                result["replayed_state_hash"] = replayed_hash
                result["plan"] = guidance
                result["extra_model_calls"] = 1 if plan_result else 0
                if plan_result:
                    result["input_tokens"] += plan_result.input_tokens
                    result["output_tokens"] += plan_result.output_tokens
                    result["model_calls"] += 1
                action_results[action] = result

            none_score = float(action_results["NONE"]["score"])
            replan_score = float(action_results["REPLAN"]["score"])
            pair_rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "seed": branch_seed,
                    "none_score": none_score,
                    "replan_score": replan_score,
                    "raw_score_advantage": replan_score - none_score,
                    "none_success": action_results["NONE"]["success"],
                    "replan_success": action_results["REPLAN"]["success"],
                }
            )
            for action in ("NONE", "REPLAN"):
                result = action_results[action]
                action_cost = args.replan_cost if action == "REPLAN" else 0.0
                run_id = (
                    f"{args.run_label}-{candidate['candidate_id']}-{action.lower()}-s{branch_seed}"
                )
                branch_store.append(
                    {
                        "run_id": run_id,
                        "baseline_run_id": candidate["baseline_run_id"],
                        "task_id": candidate["task_id"],
                        "model_id": args.model_id,
                        "prefix_id": candidate["candidate_id"],
                        "prefix_state_hash": candidate["prefix_state_hash"],
                        "replayed_state_hash": result["replayed_state_hash"],
                        "event_type": "no_progress",
                        "action": action,
                        "seed": branch_seed,
                        "terminal_utility": float(result["score"]) - action_cost,
                        "none_utility": none_score,
                        "action_advantage": float(result["score"]) - action_cost - none_score,
                        "steps": result["steps"],
                        "input_tokens": result["input_tokens"],
                        "output_tokens": result["output_tokens"],
                    }
                )
                trace_store.append(
                    {
                        "run_id": run_id,
                        "config_hash": config_hash,
                        "harness_commit": harness_commit,
                        "checkpoint_hash": checkpoint_hash,
                        "action_cost": action_cost,
                        "raw_terminal_score": result["score"],
                        "success": result["success"],
                        "extra_model_calls": result["extra_model_calls"],
                        "total_model_calls": result["model_calls"],
                        "plan": result["plan"],
                        "parser_failure": result["parser_failure"],
                        "invalid_actions": result["invalid_actions"],
                        "wall_time_seconds": result["wall_time_seconds"],
                        "termination_reason": result["termination_reason"],
                        "trace": result["trace"],
                    }
                )
            print(json.dumps(pair_rows[-1], ensure_ascii=False), flush=True)

    decision = decide_branch_pilot(
        pair_rows,
        integrity_ok=integrity_report["ok"],
        all_zero_decision="STOP_REPLAN_ON_SOURCE_POLICY_V2",
    )
    report = {
        "experiment_id": "L3-SOURCE-POLICY-V2-NONE-REPLAN-BRANCH-PILOT",
        "frozen_config": frozen_config,
        "config_hash": config_hash,
        "harness_commit": harness_commit,
        "model_load_seconds": model_load_seconds,
        "integrity": integrity_report,
        "pair_count": len(pair_rows),
        "branch_record_count": branch_store.count(),
        "pairs": pair_rows,
        "decision": decision,
        "ok": integrity_report["ok"]
        and len(pair_rows) == len(candidates) * 3
        and branch_store.count() == len(candidates) * 6,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
