#!/usr/bin/env python3
"""Run the frozen three-task Source Policy Gate v2 on local Qwen checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml
from alfworld.agents.environment import get_environment
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from traceh_core.agent import allowed_next_tokens
from traceh_core.records import AppendOnlyRecordStore
from traceh_core.schema import load_schema, validate_instance
from traceh_core.source_policy_v2 import (
    SourcePolicyDecision,
    evaluate_source_policy_gate,
    run_source_policy_episode,
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


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


def load_decider(
    model_path: Path,
    *,
    max_input_tokens: int,
    deliberation_max_tokens: int,
):
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

    def decide(
        messages: list[dict[str, str]],
        admissible_commands: Sequence[str],
    ) -> SourcePolicyDecision:
        commands = [str(command) for command in admissible_commands]
        deliberation_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        deliberation_inputs = {
            key: value.to(model.device) for key, value in deliberation_inputs.items()
        }
        deliberation_input_tokens = int(deliberation_inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            deliberation_generated = model.generate(
                **deliberation_inputs,
                max_new_tokens=deliberation_max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        deliberation_ids = deliberation_generated[0, deliberation_input_tokens:]
        deliberation = tokenizer.decode(deliberation_ids, skip_special_tokens=True).strip()

        action_messages = [
            *messages,
            {"role": "assistant", "content": deliberation},
            {
                "role": "user",
                "content": (
                    "Commit to one selectable command. Respond with exactly one line: "
                    "ACTION: <exact command>"
                ),
            },
        ]
        action_inputs = tokenizer.apply_chat_template(
            action_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        action_inputs = {key: value.to(model.device) for key, value in action_inputs.items()}
        action_input_tokens = int(action_inputs["input_ids"].shape[-1])
        completions = [
            tokenizer.encode(f"ACTION: {command}", add_special_tokens=False)
            for command in commands
        ]
        if any(not completion for completion in completions):
            raise RuntimeError("empty command completion in trie decoder")

        def prefix_allowed_tokens_fn(_batch_id: int, input_ids: torch.Tensor) -> list[int]:
            return allowed_next_tokens(
                completions,
                input_ids[action_input_tokens:].tolist(),
                eos_token_id=tokenizer.eos_token_id,
            )

        with torch.inference_mode():
            action_generated = model.generate(
                **action_inputs,
                max_new_tokens=max(len(completion) for completion in completions) + 1,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
        action_ids = action_generated[0, action_input_tokens:]
        action_output_tokens = int(action_ids.shape[-1])
        if action_output_tokens and int(action_ids[-1]) == tokenizer.eos_token_id:
            action_output_tokens -= 1
        return SourcePolicyDecision(
            raw_output=tokenizer.decode(action_ids, skip_special_tokens=True),
            deliberation=deliberation,
            input_tokens=deliberation_input_tokens + action_input_tokens,
            output_tokens=int(deliberation_ids.shape[-1]) + action_output_tokens,
            model_calls=2,
        )

    return model, decide


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--deliberation-max-tokens", type=int, default=64)
    parser.add_argument("--repeat-limit", type=int, default=2)
    parser.add_argument("--memory-limit", type=int, default=12)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument(
        "--evaluation-mode",
        choices=["qualification", "expansion", "ablation"],
        default="qualification",
    )
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    if args.evaluation_mode == "qualification" and (
        args.episodes != 3 or args.start_offset != 0
    ):
        raise ValueError("the frozen Source Policy Gate requires exactly three episodes")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    checkpoint_hash = sha256_bytes(args.checkpoint_manifest.read_bytes())
    harness_commit = git_state(args.project_root.resolve())
    frozen_config = {
        "run_label": args.run_label,
        "model_id": args.model_id,
        "checkpoint_hash": checkpoint_hash,
        "precision": "nf4-double-quant-bf16-compute",
        "information_class": "DEV_SOURCE_VALIDATION",
        "policy_id": "SOURCE_POLICY_V2",
        "seed": args.seed,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "max_input_tokens": args.max_input_tokens,
        "deliberation_max_tokens": args.deliberation_max_tokens,
        "repeat_limit": args.repeat_limit,
        "memory_limit": args.memory_limit,
        "start_offset": args.start_offset,
        "evaluation_mode": args.evaluation_mode,
        "model_calls_per_step": 2,
        "parser_contract": "exact-text-v1",
        "decoder": "deliberation-plus-command-trie-v2",
        "oracle_fields_allowed": [],
        "utility": "alfworld-score",
    }
    config_hash = canonical_hash(frozen_config)
    episode_schema = load_schema(args.project_root / "experiments/schemas/traceh-episode.schema.json")
    episode_store = AppendOnlyRecordStore(
        args.raw_root / "episodes",
        validator=lambda record: validate_instance(record, episode_schema),
    )
    trace_store = AppendOnlyRecordStore(args.raw_root / "traces")

    load_started = time.perf_counter()
    model, decide = load_decider(
        args.model.resolve(),
        max_input_tokens=args.max_input_tokens,
        deliberation_max_tokens=args.deliberation_max_tokens,
    )
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started

    report_rows = []
    for offset in range(args.start_offset, args.start_offset + args.episodes):
        run_id = f"{args.run_label}-o{offset:03d}-s{args.seed}"
        started_at = datetime.now(timezone.utc).isoformat()
        episode_started = time.perf_counter()
        env = build_env(args.config, args.data, args.seed, offset)
        try:
            result = run_source_policy_episode(
                env,
                decide=decide,
                max_steps=args.max_steps,
                repeat_limit=args.repeat_limit,
                memory_limit=args.memory_limit,
            )
        finally:
            env.close()
        wall_time = time.perf_counter() - episode_started
        trace_path = trace_store.append(
            {
                "run_id": run_id,
                "task_id": result["task_id"],
                "model_id": args.model_id,
                "checkpoint_hash": checkpoint_hash,
                "config_hash": config_hash,
                "seed": args.seed,
                "episode_offset": offset,
                "termination_reason": result["termination_reason"],
                "trace": result["trace"],
            }
        )
        summary = {
            "run_id": run_id,
            "information_class": "DEV_SOURCE_VALIDATION",
            "task_id": result["task_id"],
            "model_id": args.model_id,
            "checkpoint_hash": checkpoint_hash,
            "precision": "nf4-double-quant-bf16-compute",
            "policy_id": "SOURCE_POLICY_V2",
            "seed": args.seed,
            "success": result["success"],
            "utility": result["score"],
            "steps": result["steps"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "invalid_actions": result["invalid_actions"],
            "wall_time_seconds": wall_time,
            "infrastructure_failure": False,
            "parser_failure": result["parser_failure"],
            "started_at": started_at,
            "harness_commit": harness_commit,
            "config_hash": config_hash,
            "trace_path": str(trace_path),
            "failure_class": None,
        }
        episode_store.append(summary)
        report_rows.append(
            {
                **summary,
                "model_calls": result["model_calls"],
                "termination_reason": result["termination_reason"],
            }
        )
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "steps": result["steps"],
                    "success": result["success"],
                    "utility": result["score"],
                    "model_calls": result["model_calls"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    total_steps = sum(row["steps"] for row in report_rows)
    total_invalid = sum(row["invalid_actions"] for row in report_rows)
    parser_failure_rate = total_invalid / total_steps if total_steps else 1.0
    aggregate = {
        "episode_count": len(report_rows),
        "success_count": sum(row["success"] for row in report_rows),
        "total_environment_steps": total_steps,
        "total_model_calls": sum(row["model_calls"] for row in report_rows),
        "total_invalid_actions": total_invalid,
        "parser_failure_rate": parser_failure_rate,
        "infrastructure_failure_count": sum(
            row["infrastructure_failure"] for row in report_rows
        ),
    }
    gate = (
        evaluate_source_policy_gate(
            episode_count=aggregate["episode_count"],
            success_count=aggregate["success_count"],
            parser_failure_rate=aggregate["parser_failure_rate"],
            infrastructure_failure_count=aggregate["infrastructure_failure_count"],
        )
        if args.evaluation_mode == "qualification"
        else {
            "decision": "DESCRIPTIVE_EXPLORATION_ONLY",
            "passed": None,
            "protocol_ok": aggregate["infrastructure_failure_count"] == 0
            and aggregate["parser_failure_rate"] <= 0.02,
        }
    )
    report = {
        "experiment_id": f"L3-SOURCE-POLICY-GATE-V2-{args.model_id.upper()}",
        "frozen_config": frozen_config,
        "config_hash": config_hash,
        "harness_commit": harness_commit,
        "model_load_seconds": model_load_seconds,
        "model_memory_footprint_gib": model.get_memory_footprint() / 1024**3,
        "episodes": report_rows,
        "aggregate": aggregate,
        "gate": gate,
        "ok": gate["protocol_ok"],
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
