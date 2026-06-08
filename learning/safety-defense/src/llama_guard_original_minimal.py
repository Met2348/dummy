"""Minimal Llama Guard mechanisms from the 2023 paper.

This file uses harmless toy category labels. It demonstrates:

1. A policy taxonomy with six original Llama Guard categories.
2. Separate prompt and response classification tasks.
3. The "safe" or "unsafe\\nOi" output format.
4. 1-vs-all per-category scoring for adaptable taxonomies.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyCategory:
    code: str
    name: str
    description: str
    toy_triggers: tuple[str, ...]


@dataclass(frozen=True)
class LlamaGuardOutput:
    label: str
    categories: tuple[str, ...]
    unsafe_score: float

    def render(self) -> str:
        if self.label == "safe":
            return "safe"
        return "unsafe\n" + ",".join(self.categories)


LLAMA_GUARD_TAXONOMY: tuple[SafetyCategory, ...] = (
    SafetyCategory(
        code="O1",
        name="Violence and Hate",
        description="Violence, hateful sentiment, or discrimination risk.",
        toy_triggers=("unsafe:violence_hate",),
    ),
    SafetyCategory(
        code="O2",
        name="Sexual Content",
        description="Sexual or explicit content risk.",
        toy_triggers=("unsafe:sexual_content",),
    ),
    SafetyCategory(
        code="O3",
        name="Guns and Illegal Weapons",
        description="Illegal weapons or weapons-use risk.",
        toy_triggers=("unsafe:guns_illegal_weapons",),
    ),
    SafetyCategory(
        code="O4",
        name="Regulated or Controlled Substances",
        description="Controlled-substance policy risk.",
        toy_triggers=("unsafe:controlled_substances",),
    ),
    SafetyCategory(
        code="O5",
        name="Suicide and Self Harm",
        description="Self-harm risk.",
        toy_triggers=("unsafe:self_harm",),
    ),
    SafetyCategory(
        code="O6",
        name="Criminal Planning",
        description="Planning or assistance for criminal activity.",
        toy_triggers=("unsafe:criminal_planning", "unsafe:cybercrime"),
    ),
)


def format_taxonomy(categories: tuple[SafetyCategory, ...] = LLAMA_GUARD_TAXONOMY) -> str:
    """Render numbered policy categories for a guard task prompt."""

    lines = []
    for cat in categories:
        lines.append(f"{cat.code}: {cat.name} - {cat.description}")
    return "\n".join(lines)


def build_guard_task(
    content: str,
    task_type: str,
    categories: tuple[SafetyCategory, ...] = LLAMA_GUARD_TAXONOMY,
) -> str:
    """Build a compact prompt/response classification task."""

    if task_type not in {"prompt", "response"}:
        raise ValueError("task_type must be 'prompt' or 'response'")
    role = "User message" if task_type == "prompt" else "Assistant response"
    return "\n".join(
        [
            "Classify the following content using only this safety taxonomy.",
            format_taxonomy(categories),
            f"{role}:",
            content,
            "Return exactly 'safe' or 'unsafe' followed by category codes.",
        ]
    )


def classify_content(
    content: str,
    categories: tuple[SafetyCategory, ...] = LLAMA_GUARD_TAXONOMY,
) -> LlamaGuardOutput:
    """Toy classifier that follows the Llama Guard output format."""

    text = content.lower()
    matched = []
    for cat in categories:
        if any(trigger in text for trigger in cat.toy_triggers):
            matched.append(cat.code)
    if not matched:
        return LlamaGuardOutput(label="safe", categories=(), unsafe_score=0.0)
    score = min(1.0, 0.45 + 0.25 * len(matched))
    return LlamaGuardOutput(label="unsafe", categories=tuple(matched), unsafe_score=score)


def one_vs_all_scores(content: str) -> dict[str, float]:
    """Score each original category by running a 1-vs-all toy task."""

    scores: dict[str, float] = {}
    for cat in LLAMA_GUARD_TAXONOMY:
        out = classify_content(content, categories=(cat,))
        scores[cat.code] = out.unsafe_score
    return scores


def max_all_score(content: str) -> float:
    """Overall binary score from per-category scores, like max-all evaluation."""

    scores = one_vs_all_scores(content)
    return max(scores.values()) if scores else 0.0


def _self_test() -> int:
    safe = classify_content("benign cooking question")
    assert safe.render() == "safe"

    unsafe = classify_content("unsafe:guns_illegal_weapons evaluation item")
    assert unsafe.label == "unsafe"
    assert unsafe.categories == ("O3",)
    assert unsafe.render() == "unsafe\nO3"

    task = build_guard_task("unsafe:criminal_planning evaluation item", "prompt")
    assert "O6: Criminal Planning" in task
    assert "User message" in task

    scores = one_vs_all_scores("unsafe:criminal_planning evaluation item")
    assert scores["O6"] > 0
    assert scores["O1"] == 0
    assert max_all_score("benign") == 0.0
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"llama_guard_original_minimal.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
