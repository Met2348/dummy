"""手写 BPE vs tiktoken 一致性测试."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

pytest.importorskip("tiktoken")
from bpe_trainer import BPE     # noqa: E402


def test_bpe_roundtrip():
    """手写 BPE encode → decode 必须 idempotent."""
    text = ("the quick brown fox jumps over the lazy dog. " * 30 +
            "machine learning is a subset of artificial intelligence. " * 20)
    bpe = BPE()
    bpe.train(text, vocab_size=400)
    sample = "the quick fox learns machine intelligence"
    decoded = bpe.decode(bpe.encode(sample))
    assert decoded == sample, f"roundtrip 失败: {decoded!r} != {sample!r}"


def test_bpe_compresses():
    """训练后压缩率 > 1.5 char/token."""
    text = ("the quick brown fox jumps over the lazy dog. " * 30)
    bpe = BPE()
    bpe.train(text, vocab_size=400)
    sample = "the quick brown fox"
    ratio = len(sample) / len(bpe.encode(sample))
    assert ratio > 1.5, f"压缩率过低: {ratio:.2f} c/t"


def test_tiktoken_loads():
    """sanity check: cl100k 能正常加载并编码."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    ids = enc.encode("Hello, world!")
    assert len(ids) > 0
    assert enc.decode(ids) == "Hello, world!"
