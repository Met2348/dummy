#!/usr/bin/env python3
"""Run append-only Qwen baseline episodes in ALFWorld."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from alfworld.agents.environment import get_environment
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from traceh_core.agent import ACTION_LABELS, allowed_next_tokens
from traceh_core.baseline import GenerationResult, run_baseline_episode
from traceh_core.records import AppendOnlyRecordStore
from traceh_core.schema import load_schema, validate_instance


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


def load_generator(
    model_path: Path,
    max_input_tokens: int,
    max_new_tokens: int,
    decoder: str,
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

    label_token_ids = [
        tokenizer.encode(f" {label}", add_special_tokens=False) for label in ACTION_LABELS
    ]
    if not all(len(token_ids) == 1 for token_ids in label_token_ids):
        raise RuntimeError("Qwen tokenizer no longer maps action labels to single tokens")

    def generate(
        messages: list[dict[str, str]],
        admissible_commands: list[str],
    ) -> GenerationResult:
        if decoder == "command-trie-v1":
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
                generated_prefix = input_ids[input_tokens:].tolist()
                return allowed_next_tokens(
                    completions,
                    generated_prefix,
                    eos_token_id=tokenizer.eos_token_id,
                )

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max(len(completion) for completion in completions) + 1,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                )
            output_ids = generated[0, input_tokens:]
            output_token_count = int(output_ids.shape[-1])
            if output_token_count and int(output_ids[-1]) == tokenizer.eos_token_id:
                output_token_count -= 1
            return GenerationResult(
                raw_output=tokenizer.decode(output_ids, skip_special_tokens=True),
                input_tokens=input_tokens,
                output_tokens=output_token_count,
            )

        if decoder == "label-logit-v1":
            if len(admissible_commands) > len(ACTION_LABELS):
                raise ValueError("label decoder supports at most 52 admissible commands")
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(
                prompt + "ACTION_LABEL:",
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
            )
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            input_tokens = int(inputs["input_ids"].shape[-1])
            with torch.inference_mode():
                logits = model(**inputs, use_cache=False).logits[0, -1]
            allowed_token_ids = [token_ids[0] for token_ids in label_token_ids[: len(admissible_commands)]]
            best_index = max(
                range(len(allowed_token_ids)),
                key=lambda index: float(logits[allowed_token_ids[index]]),
            )
            return GenerationResult(
                raw_output=f"ACTION_LABEL: {ACTION_LABELS[best_index]}",
                input_tokens=input_tokens,
                output_tokens=1,
            )

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
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        output_ids = generated[0, input_tokens:]
        return GenerationResult(
            raw_output=tokenizer.decode(output_ids, skip_special_tokens=True),
            input_tokens=input_tokens,
            output_tokens=int(output_ids.shape[-1]),
        )

    return model, generate


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
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--action-protocol",
        choices=[
            "exact-text-v1",
            "indexed-v1",
            "indexed-normalized-v2",
            "label-logit-v1",
        ],
        default="exact-text-v1",
    )
    parser.add_argument(
        "--decoder",
        choices=["generate", "label-logit-v1", "command-trie-v1"],
        default="generate",
    )
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("episodes and max_steps must be positive")
    if (args.action_protocol == "label-logit-v1") != (args.decoder == "label-logit-v1"):
        raise ValueError("label-logit-v1 protocol and decoder must be selected together")
    if args.decoder == "command-trie-v1" and args.action_protocol != "exact-text-v1":
        raise ValueError("command-trie-v1 decoder requires exact-text-v1 protocol")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    checkpoint_hash = sha256_bytes(args.checkpoint_manifest.read_bytes())
    harness_commit = git_state(args.project_root.resolve())
    frozen_config = {
        "run_label": args.run_label,
        "model_id": args.model_id,
        "checkpoint_hash": checkpoint_hash,
        "precision": "nf4-double-quant-bf16-compute",
        "information_class": "DEV_SOURCE_BRANCH",
        "policy_id": "NONE",
        "seed": args.seed,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "history_limit": 4,
        "parser_contract": args.action_protocol,
        "decoder": args.decoder,
        "parser_fallback": "look-else-sorted-first",
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
    model, generate = load_generator(
        args.model.resolve(),
        args.max_input_tokens,
        args.max_new_tokens,
        args.decoder,
    )
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started

    summaries = []
    for offset in range(args.episodes):
        run_id = f"{args.run_label}-o{offset:03d}-s{args.seed}"
        started_at = datetime.now(timezone.utc).isoformat()
        episode_started = time.perf_counter()
        env = build_env(args.config, args.data, args.seed, offset)
        try:
            result = run_baseline_episode(
                env,
                generate=generate,
                max_steps=args.max_steps,
                action_protocol=args.action_protocol,
            )
        finally:
            env.close()
        wall_time = time.perf_counter() - episode_started

        trace_record = {
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
        trace_path = trace_store.append(trace_record)
        summary = {
            "run_id": run_id,
            "information_class": "DEV_SOURCE_BRANCH",
            "task_id": result["task_id"],
            "model_id": args.model_id,
            "checkpoint_hash": checkpoint_hash,
            "precision": "nf4-double-quant-bf16-compute",
            "policy_id": "NONE",
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
        summaries.append({**summary, "termination_reason": result["termination_reason"]})
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "task_id": result["task_id"],
                    "steps": result["steps"],
                    "success": result["success"],
                    "invalid_actions": result["invalid_actions"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    total_calls = sum(item["steps"] for item in summaries)
    total_invalid = sum(item["invalid_actions"] for item in summaries)
    parser_failure_rate = total_invalid / total_calls if total_calls else 1.0
    report = {
        "experiment_id": f"L2-{args.model_id.upper()}-ALFWORLD-BASELINE-MICRO",
        "frozen_config": frozen_config,
        "config_hash": config_hash,
        "harness_commit": harness_commit,
        "model_load_seconds": model_load_seconds,
        "model_memory_footprint_gib": model.get_memory_footprint() / 1024**3,
        "episodes": summaries,
        "aggregate": {
            "episode_count": len(summaries),
            "success_count": sum(item["success"] for item in summaries),
            "total_model_calls": total_calls,
            "total_invalid_actions": total_invalid,
            "parser_failure_rate": parser_failure_rate,
            "parser_gate_threshold": 0.02,
            "parser_gate_ok": parser_failure_rate <= 0.02,
            "infrastructure_failure_count": sum(
                item["infrastructure_failure"] for item in summaries
            ),
        },
        "ok": len(summaries) == args.episodes
        and all(not item["infrastructure_failure"] for item in summaries),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
