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


def demo() -> None:
    print("=== Mock 嵌入服务（sha256 确定性占位，非真语义）===")
    texts = ["hello world", "hello world", "completely different text"]
    embs = embed_batch(texts)
    print(f"嵌入 {len(texts)} 条文本, dim={len(embs[0])}")
    print(f"cos(相同文本)  = {cosine(embs[0], embs[1]):.3f}   (确定性 → 1.000)")
    print(f"cos(不同文本)  = {cosine(embs[0], embs[2]):.3f}   (hash 占位 → 近 0)")
    print("注：embed_text 用 sha256 做确定性占位向量演示服务管线，非真语义嵌入。")


if __name__ == "__main__":
    demo()
