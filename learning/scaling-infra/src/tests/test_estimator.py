"""测试 capstone estimator."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from capstone_train_estimator import TrainSpec, estimate
from scaling_laws import (chinchilla_loss, chinchilla_optimal_split,
                          over_train_split)
from parallelism_demo import dp_memory, zero3_memory, tp_memory


def test_chinchilla_ratio():
    N, D = chinchilla_optimal_split(1e22)
    assert 15 < D / N < 25


def test_over_train_ratio():
    N, D = over_train_split(1e22, ratio=200)
    assert 190 < D / N < 210


def test_estimator_7B_feasible():
    s = TrainSpec(model_size_b=7, seq_len=2048, batch=128,
                   n_token=2e9, n_gpu=8, gpu_vram_gb=80,
                   gpu_tflops=312)
    p = estimate(s)
    assert p.feasible
    assert "ZeRO" in p.strategy or "FSDP" in p.strategy
    assert p.mem_per_gpu_gb < 80


def test_estimator_70B_needs_advanced():
    s = TrainSpec(model_size_b=70, seq_len=2048, batch=256,
                   n_token=15e12, n_gpu=8, gpu_vram_gb=80,
                   gpu_tflops=312)
    p = estimate(s)
    assert p.feasible
    assert p.mem_per_gpu_gb < 80


def test_dp_memory_high():
    n = 7e9
    r = dp_memory(int(n), n_gpu=8)
    assert r["per_gpu_gb"] > 50


def test_zero3_memory_low():
    n = 7e9
    r = zero3_memory(int(n), n_gpu=8)
    assert r["per_gpu_gb"] < 15


def test_tp_memory_scales():
    n = 7e9
    r1 = tp_memory(int(n), tp_size=1)
    r8 = tp_memory(int(n), tp_size=8)
    assert r8["per_gpu_gb"] < r1["per_gpu_gb"] / 4
