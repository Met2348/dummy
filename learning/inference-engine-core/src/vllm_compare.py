"""Reference benchmark using the real vllm library, for the capstone.

`python vllm_compare.py --model Qwen/Qwen2.5-0.5B`
will spin up vllm.LLM and run the same 5 cases as mini_vllm.

Skipped automatically when vllm is unavailable so CI on Windows still works.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import List


def try_import_vllm():
    try:
        from vllm import LLM, SamplingParams
        return LLM, SamplingParams
    except ImportError:
        return None, None


CASES = [
    dict(name="short-in-short-out", n=8, p=32, m=32),
    dict(name="long-in-short-out", n=4, p=1024, m=8),
    dict(name="short-in-long-out", n=8, p=16, m=256),
    dict(name="big-batch", n=32, p=16, m=32),
    dict(name="streaming-1", n=1, p=32, m=128),
]


def run_vllm(model_name: str) -> List[dict]:
    LLM, SamplingParams = try_import_vllm()
    if LLM is None:
        print("vllm not installed; skipping reference benchmark")
        return []
    llm = LLM(model=model_name, enforce_eager=True, max_model_len=2048)
    out = []
    for case in CASES:
        prompts = ["a " * case["p"]] * case["n"]
        sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=case["m"])
        t0 = time.perf_counter()
        gen = llm.generate(prompts, sp)
        elapsed = time.perf_counter() - t0
        total_out = sum(len(g.outputs[0].token_ids) for g in gen)
        out.append({
            "case": case["name"],
            "elapsed": round(elapsed, 3),
            "tokens_out": total_out,
            "throughput": round(total_out / elapsed, 1),
        })
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--save", default="bench_vllm.json")
    args = ap.parse_args()
    res = run_vllm(args.model)
    print(json.dumps(res, indent=2))
    if res:
        with open(args.save, "w") as f:
            json.dump(res, f, indent=2)
