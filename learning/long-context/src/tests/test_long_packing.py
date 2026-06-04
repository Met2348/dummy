"""测试 packing + doc mask 正确性."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from long_data_packing import (pack_documents, make_doc_mask,
                               curriculum_lengths, packing_efficiency)


def test_pack_no_overflow():
    docs = [list(range(100)), list(range(200)), list(range(500))]
    batches = pack_documents(docs, max_len=400)
    for b in batches:
        assert sum(len(d) for d in b) <= 400


def test_doc_mask_block_diagonal():
    mask = make_doc_mask([3, 4], total_len=7)
    assert mask.shape == (7, 7)
    assert mask[0, 1].item() is True
    assert mask[3, 4].item() is True
    assert mask[2, 3].item() is False
    assert mask[0, 5].item() is False


def test_curriculum_progression():
    s1 = curriculum_lengths(1)
    s2 = curriculum_lengths(2)
    s3 = curriculum_lengths(3)
    assert s1[1] <= s2[0] * 2
    assert s2[1] <= s3[0] * 2


def test_packing_efficiency_high():
    docs = [list(range(1000)) for _ in range(8)]
    batches = pack_documents(docs, max_len=4000)
    eff = packing_efficiency(batches, max_len=4000)
    assert eff > 0.9
