"""Minimal mechanisms from the TinyStories paper.

TinyStories is not mainly a new model architecture. Its core idea is to
control the data distribution so very small language models can learn
coherent English, then evaluate generations with a teacher-style rubric
and check that outputs are not simple memorization.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import math
import re

import torch
import torch.nn.functional as F


BASIC_CHILD_VOCAB = {
    "a", "and", "apple", "bad", "ball", "big", "boy", "cat", "come",
    "dog", "door", "eat", "friend", "girl", "go", "good", "happy",
    "help", "house", "jump", "little", "look", "mom", "play", "red",
    "road", "run", "sad", "said", "see", "small", "story", "the",
    "to", "toy", "tree", "want", "was", "went", "with",
}


@dataclass(frozen=True)
class StoryConstraints:
    """Prompt controls used to make synthetic stories diverse."""

    verb: str
    noun: str
    adjective: str
    features: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelSpec:
    """Tiny GPT-Neo-like model specification used for comparisons."""

    hidden: int
    layers: int
    heads: int
    vocab_size: int = 10_000
    context_length: int = 512


def tokenize_words(text: str) -> list[str]:
    """Lowercase word tokenizer for dataset-complexity checks."""

    return re.findall(r"[a-z]+", text.lower())


def build_story_prompt(constraints: StoryConstraints) -> str:
    """Build the synthetic-data prompt style described in the paper."""

    feature_text = ", ".join(constraints.features) or "a simple plot"
    return (
        "Write a short story using only very simple words that a "
        "3 year old child would likely understand. The story should "
        f"use the verb {constraints.verb}, the noun {constraints.noun}, "
        f"and the adjective {constraints.adjective}. The story should "
        f"have these features: {feature_text}. Remember to only use "
        "simple words."
    )


def child_vocab_fraction(text: str, vocab: set[str] | None = None) -> float:
    """Fraction of tokens that fall inside a small child-style vocab."""

    vocab = BASIC_CHILD_VOCAB if vocab is None else vocab
    words = tokenize_words(text)
    if not words:
        return 0.0
    return sum(word in vocab for word in words) / len(words)


def lexical_diversity(texts: list[str]) -> float:
    """Unique-word ratio across a group of stories."""

    words = []
    for text in texts:
        words.extend(tokenize_words(text))
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def ngrams(tokens: list[str], k: int) -> Counter[tuple[str, ...]]:
    """Return k-gram counts."""

    if k <= 0:
        raise ValueError("k must be positive")
    return Counter(tuple(tokens[i:i + k]) for i in range(len(tokens) - k + 1))


def rouge_k_precision(source: str, target: str, k: int = 2) -> float:
    """Rouge-k precision: how much of source is contained in target."""

    source_grams = ngrams(tokenize_words(source), k)
    target_grams = ngrams(tokenize_words(target), k)
    total = sum(source_grams.values())
    if total == 0:
        return 0.0
    overlap = 0
    for gram, count in source_grams.items():
        overlap += min(count, target_grams.get(gram, 0))
    return overlap / total


def rouge_k_fmeasure(text_a: str, text_b: str, k: int = 2) -> float:
    """Symmetric Rouge-k f-measure."""

    p_ab = rouge_k_precision(text_a, text_b, k)
    p_ba = rouge_k_precision(text_b, text_a, k)
    if p_ab + p_ba == 0:
        return 0.0
    return 2 * p_ab * p_ba / (p_ab + p_ba)


def nearest_training_overlap(
    generation: str,
    training_stories: list[str],
    k: int = 2,
) -> float:
    """Highest Rouge-k precision to any training story."""

    if not training_stories:
        return 0.0
    return max(
        rouge_k_precision(generation, story, k)
        for story in training_stories
    )


def gpt_eval_prompt(beginning: str, completion: str) -> str:
    """Teacher-style GPT-Eval prompt from the paper."""

    return (
        "The student is given a beginning of a story and must complete it.\n"
        "Grade the completion for grammar, creativity, consistency with "
        "the beginning, and whether the plot makes sense.\n\n"
        "Beginning:\n"
        f"{beginning}\n\n"
        "Student completion:\n"
        f"{completion}\n"
    )


def toy_teacher_scores(beginning: str, completion: str) -> dict[str, float]:
    """Deterministic toy rubric for local tests.

    This is not GPT-4. It only mirrors the rubric dimensions so the
    evaluation contract is executable without an API call.
    """

    words = tokenize_words(completion)
    sentences = re.split(r"[.!?]+", completion)
    nonempty_sentences = [s.strip() for s in sentences if s.strip()]
    grammar = 4.0 + min(4.0, len(nonempty_sentences))
    creativity = 4.0 + min(3.0, lexical_diversity([completion]) * 10)
    beginning_names = {
        word for word in re.findall(r"\b[A-Z][a-z]+\b", beginning)
    }
    kept_names = sum(name.lower() in words for name in beginning_names)
    consistency = 5.0 + min(4.0, kept_names * 2.0)
    plot = 4.0 + min(4.0, len(words) / 20.0)
    return {
        "grammar": min(grammar, 10.0),
        "creativity": min(creativity, 10.0),
        "consistency": min(consistency, 10.0),
        "plot": min(plot, 10.0),
    }


def rough_gpt_params(spec: ModelSpec) -> int:
    """Rough GPT-style parameter count excluding optimizer states."""

    h = spec.hidden
    attn = 4 * h * h
    mlp = 8 * h * h
    layer_norm = 4 * h
    per_layer = attn + mlp + layer_norm
    embeddings = spec.vocab_size * h + spec.context_length * h
    return int(embeddings + spec.layers * per_layer)


def capability_profile(spec: ModelSpec) -> dict[str, float]:
    """Toy profile matching the paper's width-depth interpretation."""

    width_term = math.log2(max(spec.hidden, 64) / 64)
    depth_term = math.log2(max(spec.layers, 1))
    grammar = min(10.0, 5.0 + 0.8 * width_term + 0.3 * depth_term)
    facts = min(10.0, 3.0 + 1.2 * width_term + 0.2 * depth_term)
    context = min(10.0, 3.0 + 0.5 * width_term + 1.0 * depth_term)
    return {
        "grammar": grammar,
        "factual_knowledge": facts,
        "context_tracking": context,
    }


def next_token_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """Standard autoregressive next-token loss.

    logits: [batch, time, vocab]
    target_ids: [batch, time]
    """

    if logits.ndim != 3 or target_ids.ndim != 2:
        raise ValueError("expected logits [B,T,V] and targets [B,T]")
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        target_ids[:, 1:].reshape(-1),
    )


def _self_test() -> None:
    prompt = build_story_prompt(
        StoryConstraints(
            verb="come",
            noun="road",
            adjective="sad",
            features=("dialogue", "bad ending"),
        )
    )
    assert "come" in prompt and "sad" in prompt
    assert child_vocab_fraction("the happy dog went to the tree") == 1.0
    assert rouge_k_precision("a b c", "a b c d", k=2) == 1.0
    small = rough_gpt_params(ModelSpec(64, 2, 2))
    large = rough_gpt_params(ModelSpec(256, 8, 8))
    assert large > small


if __name__ == "__main__":
    _self_test()
    print("TinyStories original minimal checks passed")
