"""Capstone-1 - R1-tiny mock deployment.

Simulates loading a Module-4 reasoning-r1 checkpoint, quantising it via
Module-5 AWQ, and serving it through a streaming OpenAI-compatible endpoint.
The streaming generator yields a `<think>...</think><answer>...</answer>`
trace so the client can verify the thinking-trace UX without a real model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class MockR1Model:
    name: str = "r1-tiny-awq"
    quant: str = "awq-4bit"

    def stream(self, prompt: str) -> Iterable[str]:
        yield "<think>"
        for w in ("Let", "me", "solve", "this", "step", "by", "step."):
            yield w
        yield "16"
        yield "-"
        yield "3"
        yield "-"
        yield "4"
        yield "="
        yield "9"
        yield ","
        yield "and"
        yield "9"
        yield "*"
        yield "2"
        yield "="
        yield "18."
        yield "</think>"
        yield "<answer>"
        yield "18"
        yield "</answer>"


QUESTIONS = [
    "Janet's ducks lay 16 eggs/day. She eats 3, bakes 4 muffins, sells the rest at $2 each. How much/day?",
    "If x+5=12, what is x?",
    "A train travels 60 mph for 2 hours. How far?",
    "What is 7 * 8?",
    "Sum of first 10 positive integers?",
]


def run_demo() -> List[dict]:
    model = MockR1Model()
    out = []
    for q in QUESTIONS:
        full = "".join(model.stream(q))
        thinking = full.split("<answer>")[0]
        answer = full.split("<answer>")[1].split("</answer>")[0] if "<answer>" in full else ""
        out.append({"question": q, "thinking_present": "<think>" in thinking, "answer": answer.strip()})
    return out
