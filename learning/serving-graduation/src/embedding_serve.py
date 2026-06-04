"""Mock embedding server."""
from __future__ import annotations

from typing import List
import hashlib


def embed_text(text: str, dim: int = 64) -> List[float]:
    h = hashlib.sha256(text.encode()).digest()
    return [(b - 128) / 128 for b in h[:dim]]


def embed_batch(texts: List[str], dim: int = 64) -> List[List[float]]:
    return [embed_text(t, dim) for t in texts]


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / max(na * nb, 1e-9)
