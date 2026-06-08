"""Minimal mechanisms from the Vision-R1 paper.

The real paper trains Qwen2.5-VL models with a 200K cold-start CoT
dataset and GRPO plus PTST. This file keeps only the paper mechanisms
that can be inspected on a laptop:

1. Modality bridging: image-question data -> detailed text description.
2. Hard formatting result reward: reward is 1 only when format and
   answer are both correct.
3. GRPO group-relative advantage and clipped surrogate terms.
4. PTST stage schedule: short reasoning first, then longer reasoning.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import torch


@dataclass(frozen=True)
class MultimodalQA:
    """One toy multimodal math item.

    image_facts stands in for what a vision encoder or MLLM can see.
    The paper uses real images; this toy version keeps the information
    symbolic so the data pipeline is easy to test.
    """

    image_facts: tuple[str, ...]
    question: str
    answer: str


@dataclass(frozen=True)
class VisionR1ColdSample:
    """A cold-start sample after modality bridging and R1-style CoT."""

    question: str
    detailed_description: str
    response: str
    answer: str


@dataclass(frozen=True)
class PTSTStage:
    """One stage in Progressive Thinking Suppression Training."""

    name: str
    max_tokens: int
    group_size: int
    steps: int


VISION_R1_PTST_SCHEDULE: tuple[PTSTStage, ...] = (
    PTSTStage("stage1", max_tokens=4096, group_size=16, steps=100),
    PTSTStage("stage2", max_tokens=8192, group_size=8, steps=100),
)

OPTION_RE = re.compile(r"\b([A-D])\b")


def make_pseudo_cot(item: MultimodalQA) -> str:
    """Create the first MLLM pseudo-CoT used to expose visual details."""

    facts = "; ".join(item.image_facts)
    return (
        "Caption: "
        + facts
        + "\nReasoning: identify the relevant visual facts, then solve "
        + "the question step by step."
    )


def bridge_to_text_description(item: MultimodalQA, pseudo_cot: str) -> str:
    """Convert image facts into the detailed text given to DeepSeek-R1."""

    relevant = []
    pseudo_lower = pseudo_cot.lower()
    for fact in item.image_facts:
        tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9.]+", fact)]
        if any(token in pseudo_lower for token in tokens):
            relevant.append(fact)

    if not relevant:
        relevant = list(item.image_facts)

    return "Detailed image description: " + " ".join(relevant)


def ask_text_reasoner(description: str, question: str, answer: str) -> str:
    """Toy stand-in for DeepSeek-R1 producing a complex CoT."""

    return (
        "<think>Okay, let's inspect the visual facts. "
        + description
        + " The question asks: "
        + question
        + " I should verify the option before answering. "
        + "The consistent final option is "
        + answer
        + ".</think><answer>Final Answer:"
        + answer
        + "</answer>"
    )


def build_cold_start_sample(item: MultimodalQA) -> VisionR1ColdSample:
    """Run the toy Vision-R1-cold construction pipeline."""

    pseudo_cot = make_pseudo_cot(item)
    description = bridge_to_text_description(item, pseudo_cot)
    response = ask_text_reasoner(description, item.question, item.answer)
    return VisionR1ColdSample(
        question=item.question,
        detailed_description=description,
        response=response,
        answer=item.answer,
    )


def extract_final_answer(response: str) -> str | None:
    """Extract an option-style final answer from a model response."""

    answer_block = re.search(
        r"<answer>\s*(?:Final Answer:)?\s*([A-D])\s*</answer>",
        response,
        re.IGNORECASE,
    )
    if answer_block:
        return answer_block.group(1).upper()

    direct = re.search(r"Final Answer:\s*([A-D])", response, re.IGNORECASE)
    if direct:
        return direct.group(1).upper()

    matches = OPTION_RE.findall(response.upper())
    return matches[-1] if matches else None


def has_vision_r1_format(response: str) -> bool:
    """Check the paper's required think/answer output format."""

    pattern = r"^\s*<think>.+?</think>\s*<answer>.+?</answer>\s*$"
    return re.search(pattern, response, re.DOTALL) is not None


def hard_format_result_reward(response: str, gold_answer: str) -> float:
    """HFRRF: reward 1 only when format and answer are both correct."""

    if not has_vision_r1_format(response):
        return 0.0
    predicted = extract_final_answer(response)
    return 1.0 if predicted == gold_answer.upper() else 0.0


def group_relative_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """GRPO advantage from rewards inside one sampled group."""

    rewards = rewards.float()
    std = rewards.std(unbiased=False)
    if std.item() < 1e-8:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (std + 1e-8)


def grpo_clipped_surrogate_terms(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    logp_ref: torch.Tensor,
    rewards: torch.Tensor,
    clip_eps: float = 0.2,
    beta: float = 1e-2,
) -> dict[str, torch.Tensor]:
    """Compute per-sample GRPO clipped objective terms."""

    advantages = group_relative_advantages(rewards)
    ratio = torch.exp(logp_new - logp_old)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    kl = torch.exp(logp_ref - logp_new) - (logp_ref - logp_new) - 1.0
    objective_terms = torch.minimum(unclipped, clipped) - beta * kl
    return {
        "advantages": advantages,
        "ratio": ratio,
        "kl": kl,
        "objective_terms": objective_terms,
    }


def token_count(text: str) -> int:
    """Whitespace token count for PTST toy filtering."""

    return len(text.split())


def sample_group_for_stage(candidates: list[str], stage: PTSTStage) -> list[str]:
    """Apply the PTST length limit and group-size cap."""

    allowed = [text for text in candidates if token_count(text) <= stage.max_tokens]
    return allowed[: stage.group_size]


def score_stage_group(
    candidates: list[str],
    gold_answer: str,
    stage: PTSTStage,
) -> torch.Tensor:
    """Return HFRRF rewards for the group sampled at one PTST stage."""

    group = sample_group_for_stage(candidates, stage)
    return torch.tensor(
        [hard_format_result_reward(text, gold_answer) for text in group],
        dtype=torch.float32,
    )


def _self_test() -> None:
    item = MultimodalQA(
        image_facts=(
            "Triangle ABC is congruent to triangle FDE.",
            "AF is 10 units and AD is 3.5 units.",
            "The answer option for BD is A.",
        ),
        question="Find BD. Options: A 3, B 3.5, C 6, D 7.",
        answer="A",
    )
    sample = build_cold_start_sample(item)
    assert "AF is 10" in sample.detailed_description
    assert hard_format_result_reward(sample.response, "A") == 1.0
    assert hard_format_result_reward(sample.response.replace("<think>", ""), "A") == 0.0

    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    assert abs(group_relative_advantages(rewards).mean().item()) < 1e-6


if __name__ == "__main__":
    _self_test()
    print("Vision-R1 original minimal checks passed")
