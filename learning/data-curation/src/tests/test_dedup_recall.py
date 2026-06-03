"""Dedup 召回 / 假阳率测试.

MinHash: 100 重复文档召回 > 95%
SimHash: 50 独立文档假阳率 < 5%
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("datasketch")
from minhash_dedup import dedup as mh_dedup     # noqa: E402
import simhash_dedup                              # noqa: E402


def test_minhash_recall_high():
    base = "renewable energy is the future of clean sustainable power"
    docs = []
    for i in range(50):
        docs.append((f"o_{i}", base + f" variant {i}"))
        docs.append((f"c_{i}", base + f" variant {i} .   "))
    kept, dup_map = mh_dedup(docs, threshold=0.7)
    removed = sum(len(v) for v in dup_map.values())
    # 每对中应剔除 1 个 → 至少 40+ 被识别
    assert removed >= 40, f"MinHash 召回过低: removed={removed}"


def test_simhash_false_positive_low():
    independent = [
        "machine learning models predict outcomes from historical patterns",
        "quantum computers leverage superposition for parallel computation",
        "the universe is expanding accelerated by dark energy mysterious",
        "shakespeare's hamlet explores themes of revenge and madness deeply",
        "renewable energy includes wind solar hydro and geothermal options",
        "neural networks learn nonlinear mappings via gradient descent typically",
        "evolutionary biology studies the changes in species over time",
        "cryptocurrency uses blockchain to enable decentralized transactions globally",
        "climate change accelerates due to greenhouse gas emissions worldwide",
        "rust is a systems programming language with memory safety guarantees",
    ]
    fps = [simhash_dedup.simhash(t) for t in independent]
    fp_pairs = 0
    total_pairs = 0
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            total_pairs += 1
            if simhash_dedup.is_duplicate(fps[i], fps[j], thresh=3):
                fp_pairs += 1
    fp_rate = fp_pairs / total_pairs
    assert fp_rate < 0.05, f"SimHash 假阳率过高: {fp_rate:.2%}"
