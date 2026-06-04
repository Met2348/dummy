"""Pretraining 子组件测试."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from phi_tiny_model import PhiTinyConfig, PhiTiny
from init_schedule import cosine_with_warmup, wsd, mup_lr
from data_mixture import normalize, sample_source, curriculum_stage
from dataset_shards import write_shard, load_shard, ShardManager
from distillation import kd_loss, combined_loss
from common import chinchilla_optimal_tokens, estimate_flops


def test_phi_tiny_forward():
    cfg = PhiTinyConfig(n_layer=2, hidden=128, n_head=4, n_kv_head=2,
                        head_dim=32, intermediate=256, seq_len=64,
                        vocab_size=100)
    m = PhiTiny(cfg)
    x = torch.randint(0, 100, (2, 32))
    out = m(x)
    assert out.shape == (2, 32, 100)


def test_phi_tiny_param_count():
    cfg = PhiTinyConfig()
    m = PhiTiny(cfg)
    n = sum(p.numel() for p in m.parameters())
    n_no_embed = n - m.embed.weight.numel()
    assert 230e6 < n_no_embed < 320e6


def test_schedule_curves():
    assert cosine_with_warmup(0, 1000, 1e-3, warmup=100) == 0
    assert cosine_with_warmup(100, 1000, 1e-3, warmup=100) > 9e-4
    assert wsd(50, 1000, 1e-3) > 0
    assert wsd(800, 1000, 1e-3) > 0
    assert wsd(1000, 1000, 1e-3) < 1e-5 + 1e-10


def test_mup_lr_scales():
    base = mup_lr(256, 256, 1e-3)
    big = mup_lr(2048, 256, 1e-3)
    assert big < base


def test_data_mixture():
    cfg = {"web": 0.5, "code": 0.3, "math": 0.2}
    n = normalize(cfg)
    assert abs(sum(n.values()) - 1.0) < 1e-6


def test_curriculum_stage():
    assert curriculum_stage(0, 100) == "general"
    assert curriculum_stage(90, 100) == "high_quality"


def test_shard_roundtrip(tmp_path):
    ids = list(range(1000))
    p = str(tmp_path / "test.bin")
    write_shard(ids, p)
    data = load_shard(p)
    assert len(data) == 1000
    assert int(data[100]) == 100


def test_chinchilla_calc():
    assert chinchilla_optimal_tokens(1e9) == 20e9


def test_estimate_flops():
    assert estimate_flops(1e9, 1e9) == 6e18


def test_kd_loss_positive():
    s = torch.randn(4, 100)
    t = torch.randn(4, 100)
    L = kd_loss(s, t)
    assert L.item() > 0
