"""MinHash + LSH 去重 — datasketch 库 wrapper.

教学目标：
    1. char 5-gram → MinHash 签名（128 perm）
    2. LSH index 索引 → 阈值 0.7 查近邻
    3. 1 万文档 demo

运行：
    python minhash_dedup.py --demo
"""
from __future__ import annotations

import argparse
from typing import Iterable

from datasketch import MinHash, MinHashLSH


def shingles(text: str, n: int = 5) -> Iterable[str]:
    text = text.lower().replace("\n", " ")
    for i in range(len(text) - n + 1):
        yield text[i:i + n]


def minhash_sig(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in shingles(text):
        m.update(sh.encode("utf-8"))
    return m


def dedup(docs: list[tuple[str, str]], threshold: float = 0.7,
          num_perm: int = 128) -> tuple[set[str], dict[str, list[str]]]:
    """返回 (unique_ids, dup_map) where dup_map[kept_id] = [removed_ids]."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    sigs: dict[str, MinHash] = {}
    kept: set[str] = set()
    dup_map: dict[str, list[str]] = {}
    for doc_id, text in docs:
        m = minhash_sig(text, num_perm=num_perm)
        sigs[doc_id] = m
        neighbors = lsh.query(m)
        if neighbors:
            # 把当前 doc 归入第一个邻居
            keeper = neighbors[0]
            dup_map.setdefault(keeper, []).append(doc_id)
        else:
            lsh.insert(doc_id, m)
            kept.add(doc_id)
    return kept, dup_map


def run_demo() -> None:
    """100 文档 (50 个真重复 + 50 个独立) 验证召回."""
    base = "the mitochondria is the powerhouse of the cell producing atp"
    docs: list[tuple[str, str]] = []
    # 50 个原文 + 50 个近副本（修一两字）
    for i in range(50):
        docs.append((f"orig_{i}", base + f" -- variant {i}"))
        docs.append((f"copy_{i}", base + f" -- variant {i} ."))  # 末尾加个点
    # 50 个独立文档
    independent = [
        "machine learning is a subset of artificial intelligence systems today.",
        "quantum computing relies on superposition of qubits and entanglement.",
        "the great wall of china is over thirteen thousand miles long total.",
        "shakespeare wrote thirty seven plays during the elizabethan era era.",
        "renewable energy includes solar wind hydro and geothermal sources alike.",
    ]
    for i, t in enumerate(independent):
        docs.append((f"ind_{i}", t * 4))    # 重复内容以达 shingle 数量

    kept, dup_map = dedup(docs, threshold=0.7)
    print(f"input docs: {len(docs)}")
    print(f"kept:       {len(kept)}")
    print(f"removed:    {sum(len(v) for v in dup_map.values())}")
    print(f"dup_map size: {len(dup_map)}  (cluster heads)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        run_demo()


if __name__ == "__main__":
    main()
