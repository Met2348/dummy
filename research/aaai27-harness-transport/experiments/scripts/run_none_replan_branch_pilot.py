#!/usr/bin/env python3
"""Run NONE integrity and the local paired NONE/REPLAN branch kill test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from alfworld.agents.environment import get_environment
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from traceh_core.agent import allowed_next_tokens
from traceh_core.baseline import GenerationResult, _first, _normalized_info, _snapshot
from traceh_core.branch import build_replan_messages, run_episode_continuation
from traceh_core.branch_analysis import decide_branch_pilot
from traceh_core.records import AppendOnlyRecordStore
from traceh_core.schema import load_schema, validate_instance
from traceh_core.state import state_hash


OFFSET_RE = re.compile(r"-o(\d{3})-s\d+$")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_state(project_root: Path) -> str:
    commit = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(project_root), "status", "--porcelain", "--", str(project_root)],
        text=True,
    ).strip()
    return f"{commit}-dirty" if status else commit


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


def load_model(model_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.truncation_side = "left"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        quantization_config=quantization,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.eval()
    return tokenizer, model


def make_trie_generator(
    tokenizer: Any,
    model: Any,
    *,
    max_input_tokens: int,
    do_sample: bool,
    temperature: float,
    branch_seed: int,
    start_step: int,
):
    call_index = 0

    def generate(
        messages: list[dict[str, str]],
        admissible_commands: list[str],
    ) -> GenerationResult:
        nonlocal call_index
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        input_tokens = int(inputs["input_ids"].shape[-1])
        completions = [
            tokenizer.encode(f"ACTION: {command}", add_special_tokens=False)
            for command in admissible_commands
        ]
        if any(not completion for completion in completions):
            raise RuntimeError("empty command completion in trie decoder")

        def prefix_allowed_tokens_fn(_batch_id: int, input_ids: torch.Tensor) -> list[int]:
            return allowed_next_tokens(
                completions,
                input_ids[input_tokens:].tolist(),
                eos_token_id=tokenizer.eos_token_id,
            )

        action_seed = branch_seed * 100_000 + start_step + call_index + 10_000
        torch.manual_seed(action_seed)
        torch.cuda.manual_seed_all(action_seed)
        generation_kwargs = {
            "max_new_tokens": max(len(item) for item in completions) + 1,
            "do_sample": do_sample,
            "use_cache": True,
            "pad_token_id": tokenizer.eos_token_id,
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn,
        }
        if do_sample:
            generation_kwargs.update(
                temperature=temperature,
                top_p=1.0,
                top_k=0,
            )
        else:
            generation_kwargs.update(
                temperature=None,
                top_p=None,
                top_k=None,
            )
        with torch.inference_mode():
            generated = model.generate(**inputs, **generation_kwargs)
        output_ids = generated[0, input_tokens:]
        output_token_count = int(output_ids.shape[-1])
        if output_token_count and int(output_ids[-1]) == tokenizer.eos_token_id:
            output_token_count -= 1
        call_index += 1
        return GenerationResult(
            raw_output=tokenizer.decode(output_ids, skip_special_tokens=True),
            input_tokens=input_tokens,
            output_tokens=output_token_count,
        )

    return generate


def generate_plan(
    tokenizer: Any,
    model: Any,
    messages: list[dict[str, str]],
    *,
    seed: int,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
) -> GenerationResult:
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    input_tokens = int(inputs["input_ids"].shape[-1])
    plan_seed = 900_000 + seed
    torch.manual_seed(plan_seed)
    torch.cuda.manual_seed_all(plan_seed)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            top_k=0,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    output_ids = generated[0, input_tokens:]
    return GenerationResult(
        raw_output=tokenizer.decode(output_ids, skip_special_tokens=True),
        input_tokens=input_tokens,
        output_tokens=int(output_ids.shape[-1]),
    )


def restore_prefix(
    candidate: dict[str, Any],
    trace_record: dict[str, Any],
    *,
    config_path: Path,
    data_dir: Path,
    env_seed: int,
):
    match = OFFSET_RE.search(candidate["baseline_run_id"])
    if match is None:
        raise ValueError(f"cannot recover offset from {candidate['baseline_run_id']}")
    env = build_env(config_path, data_dir, env_seed, int(match.group(1)))
    observations, info_batch = env.reset()
    task = str(_first(observations))
    observation = task
    info = _normalized_info(info_batch)
    history: list[tuple[str, str]] = []
    score = 0.0
    for step_index in range(int(candidate["step_index"])):
        action = trace_record["trace"][step_index]["selected_action"]
        observations, scores, dones, info_batch = env.step([action])
        observation = str(_first(observations))
        score = float(_first(scores))
        info = _normalized_info(info_batch)
        history.append((action, observation))
        if bool(_first(dones)):
            env.close()
            raise RuntimeError("baseline terminated before selected prefix")
    replayed_state = _snapshot(
        observation,
        info,
        int(candidate["step_index"]),
        score,
    )
    replayed_hash = state_hash(replayed_state)
    return env, task, observation, info, history, score, replayed_hash


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
    parser.add_argument("--integrity-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--env-seed", type=int, default=20260712)
    parser.add_argument("--max-total-steps", type=int, default=50)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--replan-max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--replan-cost", type=float, default=0.03)
    parser.add_argument(
        "--all-zero-decision",
        default="MOVE_TO_QWEN3_8B",
        help="Decision label emitted when every paired raw terminal score is zero.",
    )
    args = parser.parse_args()

    candidate_report = json.loads(args.candidates.read_text(encoding="utf-8"))
    candidates = candidate_report["candidates"]
    checkpoint_hash = hashlib.sha256(args.checkpoint_manifest.read_bytes()).hexdigest()
    harness_commit = git_state(args.project_root.resolve())
    frozen_config = {
        "run_label": args.run_label,
        "model_id": args.model_id,
        "checkpoint_hash": checkpoint_hash,
        "precision": "nf4-double-quant-bf16-compute",
        "decoder": "command-trie-v1",
        "parser": "exact-text-v1",
        "env_seed": args.env_seed,
        "max_total_steps": args.max_total_steps,
        "max_input_tokens": args.max_input_tokens,
        "temperature": args.temperature,
        "branch_seeds": [0, 1, 2],
        "repeat_candidate_count": 2,
        "replan_max_tokens": args.replan_max_tokens,
        "replan_extra_calls": 1,
        "replan_cost": args.replan_cost,
    }
    config_hash = canonical_hash(frozen_config)
    load_started = time.perf_counter()
    tokenizer, model = load_model(args.model.resolve())
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started

    trace_cache = {
        candidate["candidate_id"]: json.loads(Path(candidate["trace_path"]).read_text(encoding="utf-8"))
        for candidate in candidates
    }

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
            generate = make_trie_generator(
                tokenizer,
                model,
                max_input_tokens=args.max_input_tokens,
                do_sample=False,
                temperature=args.temperature,
                branch_seed=0,
                start_step=int(candidate["step_index"]),
            )
            result = run_episode_continuation(
                env,
                task=task,
                observation=observation,
                info=info,
                history=history,
                score=score,
                start_step=int(candidate["step_index"]),
                max_total_steps=args.max_total_steps,
                generate=generate,
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
            "terminal_score": result["score"],
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
        "experiment_id": "L3-NONE-CONTINUATION-INTEGRITY",
        "candidate_count": len(integrity_rows),
        "matching_continuations": sum(item["ok"] for item in integrity_rows),
        "rows": integrity_rows,
        "ok": bool(integrity_rows) and all(item["ok"] for item in integrity_rows),
    }
    args.integrity_report.parent.mkdir(parents=True, exist_ok=True)
    args.integrity_report.write_text(
        json.dumps(integrity_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not integrity_report["ok"]:
        print(json.dumps(integrity_report, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    branch_schema = load_schema(args.project_root / "experiments/schemas/traceh-branch.schema.json")
    branch_store = AppendOnlyRecordStore(
        args.raw_root / "branches",
        validator=lambda record: validate_instance(record, branch_schema),
    )
    trace_store = AppendOnlyRecordStore(args.raw_root / "traces")
    pair_rows = []

    for candidate_index, candidate in enumerate(candidates):
        seeds = [0, 1, 2] if candidate_index < 2 else [0]
        trace_record = trace_cache[candidate["candidate_id"]]
        for branch_seed in seeds:
            action_results = {}
            for action in ("NONE", "REPLAN"):
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
                    plan_messages = build_replan_messages(
                        task=task,
                        observation=observation,
                        history=history,
                        admissible_commands=sorted(
                            str(item) for item in info.get("admissible_commands", [])
                        ),
                    )
                    plan_result = generate_plan(
                        tokenizer,
                        model,
                        plan_messages,
                        seed=branch_seed,
                        max_input_tokens=args.max_input_tokens,
                        max_new_tokens=args.replan_max_tokens,
                        temperature=args.temperature,
                    )
                    guidance = plan_result.raw_output
                generate = make_trie_generator(
                    tokenizer,
                    model,
                    max_input_tokens=args.max_input_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    branch_seed=branch_seed,
                    start_step=int(candidate["step_index"]),
                )
                started = time.perf_counter()
                try:
                    result = run_episode_continuation(
                        env,
                        task=task,
                        observation=observation,
                        info=info,
                        history=history,
                        score=score,
                        start_step=int(candidate["step_index"]),
                        max_total_steps=args.max_total_steps,
                        generate=generate,
                        guidance=guidance,
                    )
                finally:
                    env.close()
                result["wall_time_seconds"] = time.perf_counter() - started
                result["replayed_state_hash"] = replayed_hash
                result["plan"] = plan_result.raw_output if plan_result else None
                result["extra_model_calls"] = 1 if plan_result else 0
                result["input_tokens"] += plan_result.input_tokens if plan_result else 0
                result["output_tokens"] += plan_result.output_tokens if plan_result else 0
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
                terminal_utility = float(result["score"]) - action_cost
                run_id = (
                    f"{args.run_label}-{candidate['candidate_id']}-{action.lower()}-s{branch_seed}"
                )
                branch_record = {
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
                    "terminal_utility": terminal_utility,
                    "none_utility": none_score,
                    "action_advantage": terminal_utility - none_score,
                    "steps": result["steps"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                }
                trace_payload = {
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "harness_commit": harness_commit,
                    "checkpoint_hash": checkpoint_hash,
                    "action_cost": action_cost,
                    "raw_terminal_score": result["score"],
                    "success": result["success"],
                    "extra_model_calls": result["extra_model_calls"],
                    "plan": result["plan"],
                    "parser_failure": result["parser_failure"],
                    "invalid_actions": result["invalid_actions"],
                    "wall_time_seconds": result["wall_time_seconds"],
                    "termination_reason": result["termination_reason"],
                    "trace": result["trace"],
                }
                trace_store.append(trace_payload)
                branch_store.append(branch_record)
            print(
                json.dumps(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "seed": branch_seed,
                        "none_score": none_score,
                        "replan_score": replan_score,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    decision = decide_branch_pilot(
        pair_rows,
        integrity_ok=integrity_report["ok"],
        all_zero_decision=args.all_zero_decision,
    )
    expected_pair_count = len(candidates) + 2 * min(2, len(candidates))
    expected_branch_record_count = expected_pair_count * 2
    report = {
        "experiment_id": "L3-NONE-REPLAN-BRANCH-KILL-TEST",
        "frozen_config": frozen_config,
        "config_hash": config_hash,
        "harness_commit": harness_commit,
        "model_load_seconds": model_load_seconds,
        "integrity": {
            "candidate_count": integrity_report["candidate_count"],
            "matching_continuations": integrity_report["matching_continuations"],
            "ok": integrity_report["ok"],
        },
        "pair_count": len(pair_rows),
        "branch_record_count": branch_store.count(),
        "pairs": pair_rows,
        "decision": decision,
        "ok": integrity_report["ok"]
        and len(pair_rows) == expected_pair_count
        and branch_store.count() == expected_branch_record_count,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
