"""SimHash 64-bit 手写实现 — Charikar 2002.

教学目标：
    1. 实现 64-bit SimHash fingerprint
    2. Hamming 距离比较
    3. 与 cosine 相似度的等价关系

运行：
    python simhash_dedup.py --demo
"""
from __future__ import annotations

import argparse
import hashlib
import re
from typing import Iterable


def _tokens(text: str) -> Iterable[str]:
    return re.findall(r"\w+", text.lower())


def _hash_token(t: str, dim: int = 64) -> int:
    """SHA-1 截到 dim bit."""
    h = hashlib.sha1(t.encode("utf-8")).digest()
    n = int.from_bytes(h[: (dim + 7) // 8], "big")
    return n & ((1 << dim) - 1)


def simhash(text: str, dim: int = 64) -> int:
    """64-bit SimHash fingerprint."""
    counts = [0] * dim
    for t in _tokens(text):
        h = _hash_token(t, dim)
        for i in range(dim):
            counts[i] += 1 if (h >> i) & 1 else -1
    fp = 0
    for i in range(dim):
        if counts[i] > 0:
            fp |= 1 << i
    return fp


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def is_duplicate(a: int, b: int, thresh: int = 3) -> bool:
    return hamming(a, b) <= thresh


def run_demo() -> None:
    pairs = [
        ("the cat sits on the mat",
         "the cat sat on the mat"),                # near-dup
        ("the cat sits on the mat",
         "machine learning is artificial intelligence"),   # diff
        ("python is a programming language",
         "python is a great programming language"),         # near-dup
        ("renewable energy includes solar wind",
         "renewable energy includes solar wind power"),     # near-dup
    ]
    for a, b in pairs:
        fa, fb = simhash(a), simhash(b)
        d = hamming(fa, fb)
        print(f"  hamming={d:2d}  dup={is_duplicate(fa, fb)}  | "
              f"A: {a[:30]!r}  vs  B: {b[:30]!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        run_demo()


if __name__ == "__main__":
    main()
