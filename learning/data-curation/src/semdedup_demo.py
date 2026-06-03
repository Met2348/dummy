"""SemDeDup 教学版 — sentence-transformer embedding + cosine.

教学目标：
    1. 用 SentenceTransformer 算 doc embedding
    2. 簇内两两 cosine ≥ 0.95 视为 dup
    3. 处理 paraphrase 类近重复

运行：
    python semdedup_demo.py --demo
注意：首次会下载 ~80MB 的 MiniLM 模型。
"""
from __future__ import annotations

import argparse

import numpy as np


def embed(texts: list[str]):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return emb


def cosine_matrix(emb: np.ndarray) -> np.ndarray:
    # already normalized → dot product == cosine
    return emb @ emb.T


def dedup(texts: list[str], thresh: float = 0.95) -> tuple[list[int], list[tuple[int, int, float]]]:
    """返回 (kept_idx, removed_pairs[(removed_i, kept_j, sim)])."""
    emb = embed(texts)
    sim = cosine_matrix(emb)
    n = len(texts)
    kept: list[int] = []
    removed: list[tuple[int, int, float]] = []
    is_dup = [False] * n
    for i in range(n):
        if is_dup[i]:
            continue
        kept.append(i)
        for j in range(i + 1, n):
            if not is_dup[j] and sim[i, j] >= thresh:
                is_dup[j] = True
                removed.append((j, i, float(sim[i, j])))
    return kept, removed


def run_demo() -> None:
    texts = [
        "The cat is on the mat.",
        "A feline rests upon the rug.",           # paraphrase of 0
        "Python is a programming language.",
        "Python is a coding language.",            # paraphrase of 2
        "Quantum computing uses qubits.",
        "The mitochondria produces ATP.",
    ]
    kept, removed = dedup(texts, thresh=0.85)     # 教学放宽到 0.85
    print("Kept indices:", kept)
    for i, j, s in removed:
        print(f"  removed [{i}] ≈ [{j}]  sim={s:.3f}")
        print(f"    [{i}] {texts[i]}")
        print(f"    [{j}] {texts[j]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        run_demo()


if __name__ == "__main__":
    main()
