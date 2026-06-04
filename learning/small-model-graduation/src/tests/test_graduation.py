"""Graduation capstone 测试."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from vanilla_gpt2 import GPT2Config, VanillaGPT2
from train_variant import VARIANTS
from bench_matrix import EXPECTED, METRICS, ablation_breakdown
from common import CKPT_VARIANTS, variant_desc


def test_vanilla_gpt2_forward():
    c = GPT2Config(vocab_size=100, hidden=64, n_head=4,
                    n_layer=2, seq_len=64)
    m = VanillaGPT2(c)
    x = torch.randint(0, 100, (2, 32))
    out = m(x)
    assert out.shape == (2, 32, 100)


def test_vanilla_gpt2_params():
    c = GPT2Config()
    m = VanillaGPT2(c)
    n = sum(p.numel() for p in m.parameters())
    n_no_embed = n - m.tok_embed.weight.numel()
    assert 90e6 < n_no_embed < 200e6


def test_all_variants_configured():
    for v in CKPT_VARIANTS:
        assert v in VARIANTS
        cfg = VARIANTS[v]
        assert cfg.max_step > 0
        assert cfg.base_lr > 0


def test_variant_progression():
    """E should have lowest val_loss; A highest."""
    losses = {v: EXPECTED[v]["val_loss"] for v in CKPT_VARIANTS}
    assert losses["A"] > losses["B"] > losses["C"] >= losses["E"]


def test_niah_only_d_e_high():
    assert EXPECTED["A"]["niah_8k"] == 0
    assert EXPECTED["B"]["niah_8k"] == 0
    assert EXPECTED["D"]["niah_8k"] > 0.5
    assert EXPECTED["E"]["niah_8k"] > 0.5


def test_ablation_data_contributes_hellaswag():
    abl = ablation_breakdown(EXPECTED)
    assert abl["data (A→B)"]["hellaswag"] >= 0.04


def test_ablation_long_ctx_only_niah():
    abl = ablation_breakdown(EXPECTED)
    diff = abl["long_ctx (C→D)"]
    assert diff["niah_8k"] >= 0.5
    assert abs(diff["hellaswag"]) < 0.05


def test_variant_descriptions():
    for v in CKPT_VARIANTS:
        assert len(variant_desc(v)) > 5
