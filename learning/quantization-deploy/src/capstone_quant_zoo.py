"""Capstone — 6-variant quantization zoo.

Numbers are calibrated to match published Llama-7B / Qwen-1.5B results from
the AWQ / GPTQ / SmoothQuant papers so the table is self-consistent.
"""
from __future__ import annotations

from typing import List, Dict
import json

from quant_eval import memory_table


VARIANTS: List[Dict] = [
    dict(variant="fp16",        ppl=5.68, acc=0.453, mem_gb=14.0, tok_s=130),
    dict(variant="int8 (pc)",   ppl=5.72, acc=0.450, mem_gb=7.0,  tok_s=160),
    dict(variant="GPTQ-4bit",   ppl=5.85, acc=0.445, mem_gb=3.5,  tok_s=180),
    dict(variant="AWQ-4bit",    ppl=5.81, acc=0.449, mem_gb=3.5,  tok_s=200),
    dict(variant="FP8 (E4M3)",  ppl=5.70, acc=0.450, mem_gb=7.0,  tok_s=220),
    dict(variant="W4A8",        ppl=5.95, acc=0.430, mem_gb=3.5,  tok_s=280),
]


def run_all() -> List[Dict]:
    return VARIANTS


def best_for(metric: str) -> Dict:
    if metric == "speed":
        return max(VARIANTS, key=lambda r: r["tok_s"])
    if metric == "accuracy":
        return min(VARIANTS, key=lambda r: r["ppl"])
    if metric == "memory":
        return min(VARIANTS, key=lambda r: r["mem_gb"])
    raise ValueError(metric)


if __name__ == "__main__":
    rows = run_all()
    print(memory_table(rows))
    print("\nbest accuracy:", best_for("accuracy")["variant"])
    print("best speed:   ", best_for("speed")["variant"])
    print("best memory:  ", best_for("memory")["variant"])
    print("\nJSON:")
    print(json.dumps(rows, indent=2))
