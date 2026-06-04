"""Run the same question through all 5 mock ckpts and build a comparison."""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

from .ckpts import load_all, QUESTION


def run_compare() -> Dict:
    ckpts = load_all()
    rows = []
    for key, c in ckpts.items():
        rows.append({
            "ckpt": key,
            "name": c.name,
            "size_mb": c.size_mb,
            "latency_ms": c.latency_ms,
            "reasoning": c.reasoning_quality,
            "correct": c.correct,
            "response": c.response,
        })
    return {"question": QUESTION, "results": rows}


def to_md(report: Dict) -> str:
    rows = report["results"]
    md = ["## Q\n", report["question"], "", "## Responses\n"]
    md.append("| ckpt | reasoning | correct | latency | response |")
    md.append("|---|---|---|---|---|")
    for r in rows:
        ok = "OK" if r["correct"] else "NO"
        resp = r["response"].replace("\n", " ")
        md.append(f"| {r['ckpt']} | {r['reasoning']} | {ok} | {r['latency_ms']}ms | {resp} |")
    md.append("")
    md.append("## Ablation contributions")
    md.append("- vanilla -> LoRA: +correctness via PEFT")
    md.append("- LoRA -> DPO: +explicit step-by-step reasoning")
    md.append("- DPO -> R1-Zero: +<think> trace + verified answer")
    md.append("- Phi-tiny: data + arch baseline that already reasons cleanly")
    return "\n".join(md)
