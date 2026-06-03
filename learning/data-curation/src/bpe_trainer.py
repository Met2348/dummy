"""手写 BPE trainer — Karpathy minbpe 风格.

教学目标：
    1. 实现 byte-level BPE 训练算法
    2. 100 行内完成
    3. 与 tiktoken 概念对照

运行：
    python bpe_trainer.py --demo --vocab-size 300
"""
from __future__ import annotations

import argparse
import collections
from typing import Iterable


def _get_pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = collections.Counter()
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPE:
    def __init__(self):
        self.merges: list[tuple[tuple[int, int], int]] = []
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int = 512, verbose: bool = False) -> None:
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))
        num_merges = vocab_size - 256
        for i in range(num_merges):
            counts = _get_pair_counts(ids)
            if not counts:
                break
            pair = max(counts, key=counts.get)
            new_id = 256 + i
            ids = _merge(ids, pair, new_id)
            self.merges.append((pair, new_id))
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and (i + 1) % 100 == 0:
                print(f"  merge {i+1}: {pair} → {new_id} (count={counts[pair]})")

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        # 按 merge 顺序合并（与训练同序）
        for pair, new_id in self.merges:
            ids = _merge(ids, pair, new_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        text_bytes = b"".join(self.vocab[i] for i in ids)
        return text_bytes.decode("utf-8", errors="replace")


def run_demo(vocab_size: int = 300) -> None:
    text = ("the quick brown fox jumps over the lazy dog. "
            "the quick brown fox is quick and brown. "
            "machine learning is the subset of artificial intelligence. "
            "machine learning models learn patterns from data. ") * 50
    bpe = BPE()
    bpe.train(text, vocab_size=vocab_size, verbose=False)
    print(f"vocab size: {len(bpe.vocab)}")
    print(f"merge count: {len(bpe.merges)}")

    sample = "the quick brown fox"
    ids = bpe.encode(sample)
    decoded = bpe.decode(ids)
    print(f"\nsample: {sample!r}")
    print(f"bytes:  {len(sample.encode('utf-8'))}")
    print(f"tokens: {len(ids)}  → {ids[:10]}...")
    print(f"decode: {decoded!r}")
    assert decoded == sample, "encode/decode roundtrip 失败"

    # 几个 merge 可视化
    print("\nFirst 5 merges:")
    for pair, new_id in bpe.merges[:5]:
        a = bpe.vocab[pair[0]].decode("utf-8", errors="replace")
        b = bpe.vocab[pair[1]].decode("utf-8", errors="replace")
        c = bpe.vocab[new_id].decode("utf-8", errors="replace")
        print(f"  ({a!r} + {b!r}) → {c!r}  [id {new_id}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--vocab-size", type=int, default=300)
    args = ap.parse_args()
    if args.demo:
        run_demo(args.vocab_size)


if __name__ == "__main__":
    main()
