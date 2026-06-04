"""Data mixture & curriculum 工具."""
from __future__ import annotations

import random
from typing import Iterable


CONFIGS = {
    "phi": {"web": 0.5, "code": 0.2, "math": 0.1, "books": 0.1, "wiki": 0.1},
    "llama3": {"web": 0.70, "code": 0.20, "math": 0.04, "books": 0.04,
               "wiki": 0.02},
    "qwen": {"web_en": 0.4, "web_zh": 0.3, "code": 0.15, "math": 0.05,
              "books": 0.1},
    "deepseek": {"web_en": 0.3, "web_zh": 0.3, "code": 0.2, "math": 0.1,
                  "other": 0.1},
}


def sample_source(mixture: dict, rng: random.Random) -> str:
    """根据 mixture 概率采样一个 source."""
    keys = list(mixture.keys())
    weights = [mixture[k] for k in keys]
    return rng.choices(keys, weights=weights, k=1)[0]


def normalize(mixture: dict) -> dict:
    total = sum(mixture.values())
    return {k: v / total for k, v in mixture.items()}


def curriculum_stage(step: int, total_step: int) -> str:
    """Phi 风格: 80% general + 20% high-quality."""
    if step < 0.8 * total_step:
        return "general"
    return "high_quality"


def curriculum_mixture(stage: str) -> dict:
    if stage == "general":
        return normalize({"web": 0.7, "code": 0.2, "books": 0.05,
                          "wiki": 0.05})
    return normalize({"code": 0.3, "math": 0.3, "books": 0.2, "wiki": 0.2})


def wsd_annealing_mixture(step: int, total_step: int) -> dict:
    """WSD final decay phase: 注入高质."""
    if step > 0.8 * total_step:
        return normalize({"books": 0.5, "wiki": 0.2, "math": 0.3})
    return CONFIGS["phi"]


if __name__ == "__main__":
    rng = random.Random(42)
    for cfg_name in CONFIGS:
        print(f"\n=== {cfg_name} ===")
        counts = {}
        for _ in range(10000):
            s = sample_source(CONFIGS[cfg_name], rng)
            counts[s] = counts.get(s, 0) + 1
        for k, v in counts.items():
            print(f"  {k:10s}: {v/10000:.1%}")
