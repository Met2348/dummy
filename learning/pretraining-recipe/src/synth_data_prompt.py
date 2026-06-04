"""合成数据 prompt 生成器 (Phi 风格)."""
from __future__ import annotations

import random


TOPICS = [
    "neural networks", "calculus chain rule", "linked lists",
    "DNA replication", "Roman empire", "linear algebra eigenvalues",
    "cooking sourdough", "quantum entanglement",
]
AUDIENCES = ["high school student", "undergraduate", "engineer",
              "10 year old", "expert in another field"]
STYLES = ["textbook", "tutorial", "Q&A", "story", "Socratic dialogue"]


def make_prompt(topic: str, audience: str, style: str,
                 n_words: int = 500) -> str:
    return (f"You are an expert science writer. Write a clear, accurate "
            f"explanation of {topic} suitable for a {audience}. "
            f"Style: {style}. Length: about {n_words} words. "
            f"Output only the explanation, no preamble.")


def generate_seed_pool(n: int = 1000, rng=None) -> list:
    rng = rng or random.Random(42)
    pool = []
    for _ in range(n):
        topic = rng.choice(TOPICS)
        aud = rng.choice(AUDIENCES)
        style = rng.choice(STYLES)
        pool.append(make_prompt(topic, aud, style))
    return pool


def filter_quality(text: str, min_words: int = 100,
                    bad_phrases: list = None) -> bool:
    if len(text.split()) < min_words:
        return False
    bad_phrases = bad_phrases or [
        "As an AI", "I'm just an AI", "I cannot",
        "I don't know", "Sorry, I cannot",
    ]
    for b in bad_phrases:
        if b in text:
            return False
    return True


def dedup_by_prefix(samples: list, prefix_len: int = 100) -> list:
    seen = set()
    unique = []
    for s in samples:
        key = s[:prefix_len]
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    return unique


if __name__ == "__main__":
    pool = generate_seed_pool(n=10)
    for i, p in enumerate(pool[:3]):
        print(f"[{i}] {p[:200]}\n")

    print("\n=== Filter test ===")
    samples = [
        "Neural networks consist of layers " * 30,
        "Sorry, I cannot help with that.",
        "AI " * 200,
    ]
    for s in samples:
        ok = filter_quality(s)
        print(f"  {ok}: {s[:60]!r}")
