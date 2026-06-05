"""Common dataclasses & helpers for agent-memory-context."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import re
import time


def hash_embed(text: str, dim: int = 32) -> list[float]:
    vec = [0.0] * dim
    for tok in re.findall(r"\w+", text.lower()):
        vec[hash(tok) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("dim mismatch")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


_MOCK_TIME = [1717545600.0]  # 2024-06-04 deterministic baseline


def mock_now() -> float:
    _MOCK_TIME[0] += 1.0
    return _MOCK_TIME[0]


def reset_mock_time() -> None:
    _MOCK_TIME[0] = 1717545600.0


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _self_test() -> None:
    v = hash_embed("hello")
    assert len(v) == 32
    assert abs(sum(x * x for x in v) - 1.0) < 1e-6

    assert cosine([1, 0], [1, 0]) > 0.99
    assert cosine([1, 0], [0, 1]) < 0.01

    reset_mock_time()
    t1 = mock_now()
    t2 = mock_now()
    assert t2 == t1 + 1.0

    assert tokenize("Hello, World 123!") == ["hello", "world", "123"]
    print("[OK] common._self_test passed")


if __name__ == "__main__":
    _self_test()
