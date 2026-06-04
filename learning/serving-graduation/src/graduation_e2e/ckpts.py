"""5 mock checkpoints for the graduation capstone.

Each mock represents a checkpoint produced by a different prior module:
  - vanilla   : GPT-2 base (no fine-tuning), Module 1 baseline
  - lora      : LoRA-tuned (Module 1)
  - dpo       : DPO-aligned (Module 4)
  - r1_zero   : R1-Zero reasoning (Module 4)
  - phi_tiny  : Phi-tiny 270M (Module 3)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass
class Ckpt:
    name: str
    size_mb: int
    latency_ms: int
    reasoning_quality: str          # "none" / "brief" / "yes" / "strong" / "clean"
    correct: bool
    response: str


# Pre-baked outputs for the canonical Janet question
QUESTION = (
    "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes "
    "muffins with 4. She sells the remainder at $2 per egg. How much does "
    "she make per day?"
)


CKPTS: Dict[str, Ckpt] = {
    "vanilla": Ckpt(
        name="vanilla-gpt2-124m",
        size_mb=500,
        latency_ms=30,
        reasoning_quality="none",
        correct=False,
        response="$10",
    ),
    "lora": Ckpt(
        name="gpt2-lora-tuned",
        size_mb=520,
        latency_ms=35,
        reasoning_quality="brief",
        correct=True,
        response="16-3-4=9, 9*$2=$18",
    ),
    "dpo": Ckpt(
        name="gpt2-dpo-aligned",
        size_mb=520,
        latency_ms=40,
        reasoning_quality="yes",
        correct=True,
        response="Let me work through this step by step. 16-3=13, 13-4=9, 9*$2=$18",
    ),
    "r1_zero": Ckpt(
        name="gpt2-r1-zero",
        size_mb=510,
        latency_ms=80,
        reasoning_quality="strong",
        correct=True,
        response=(
            "<think>16 - 3 (breakfast) - 4 (muffins) = 9 eggs. "
            "9 * $2 = $18.</think><answer>$18</answer>"
        ),
    ),
    "phi_tiny": Ckpt(
        name="phi-tiny-270m",
        size_mb=540,
        latency_ms=60,
        reasoning_quality="clean",
        correct=True,
        response="She has 16-3-4=9 eggs left. 9 * $2 = $18.",
    ),
}


def load_all() -> Dict[str, Ckpt]:
    return CKPTS


def list_names() -> List[str]:
    return list(CKPTS.keys())
