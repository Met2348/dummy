#!/usr/bin/env python3
"""Measure local 4-bit Qwen inference readiness across several context lengths."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import accelerate
import bitsandbytes as bnb
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


GIB = 1024**3


def gib(value: int) -> float:
    return round(value / GIB, 4)


def make_inputs(tokenizer: Any, target_tokens: int) -> dict[str, torch.Tensor]:
    instruction = "Write a concise readiness note about reproducible agent experiments."

    def encode(repetitions: int) -> dict[str, torch.Tensor]:
        filler = " evidence" * repetitions
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": filler + "\n" + instruction}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )

    low, high = 0, max(16, target_tokens * 2)
    while encode(high)["input_ids"].shape[-1] < target_tokens:
        high *= 2
    while low + 1 < high:
        middle = (low + high) // 2
        if encode(middle)["input_ids"].shape[-1] <= target_tokens:
            low = middle
        else:
            high = middle
    candidates = [encode(low), encode(high)]
    return min(
        candidates,
        key=lambda item: abs(item["input_ids"].shape[-1] - target_tokens),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model_path = args.model.resolve()
    model_id = model_path.name
    weight_files = sorted(model_path.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"no safetensors weights under {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        quantization_config=quantization,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    quantized_modules = sum(
        isinstance(module, bnb.nn.Linear4bit) for module in model.modules()
    )

    runs = []
    for target_tokens in args.contexts:
        inputs = make_inputs(tokenizer, target_tokens)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        input_tokens = int(inputs["input_ids"].shape[-1])
        for repetition in range(args.repeats):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            output_ids = generated[0, input_tokens:]
            output_tokens = int(output_ids.shape[-1])
            runs.append(
                {
                    "target_context_tokens": target_tokens,
                    "input_tokens": input_tokens,
                    "repetition": repetition,
                    "output_tokens": output_tokens,
                    "elapsed_seconds": round(elapsed, 4),
                    "end_to_end_output_tokens_per_second": round(output_tokens / elapsed, 4),
                    "peak_allocated_gib": gib(torch.cuda.max_memory_allocated()),
                    "peak_reserved_gib": gib(torch.cuda.max_memory_reserved()),
                    "output": tokenizer.decode(output_ids, skip_special_tokens=True),
                    "ok": output_tokens > 0,
                }
            )

    properties = torch.cuda.get_device_properties(0)
    return {
        "experiment_id": f"L0-{model_id.upper()}-NF4-PROBE",
        "model_id": model_id,
        "seed": args.seed,
        "model_path": str(model_path),
        "weight_file_count": len(weight_files),
        "weight_bytes": sum(path.stat().st_size for path in weight_files),
        "quantization": {
            "bits": 4,
            "type": "nf4",
            "double_quant": True,
            "compute_dtype": "bfloat16",
            "linear4bit_module_count": quantized_modules,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "bitsandbytes": bnb.__version__,
            "gpu": properties.name,
            "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_total_gib": gib(properties.total_memory),
        },
        "load_seconds": round(load_seconds, 4),
        "model_memory_footprint_gib": gib(model.get_memory_footprint()),
        "allocated_after_load_gib": gib(torch.cuda.memory_allocated()),
        "runs": runs,
        "ok": quantized_modules > 0 and len(runs) == len(args.contexts) * args.repeats and all(
            item["ok"] for item in runs
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 2048, 4096])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    try:
        report = run(args)
    except Exception as error:
        report = {
            "experiment_id": f"L0-{args.model.name.upper()}-NF4-PROBE",
            "model_id": args.model.name,
            "error_type": type(error).__name__,
            "error": str(error),
            "ok": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
