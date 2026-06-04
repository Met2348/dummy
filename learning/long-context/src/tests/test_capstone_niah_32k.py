"""Capstone smoke test: 不真训, 只验 YaRN scale 与 attn_temp 公式正确."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
from capstone_yarn_llama32 import yarn_inv_freq, attn_temperature, curriculum_max_len


def test_yarn_inv_freq_shape():
    inv = yarn_inv_freq(dim=128, scale=4.0)
    assert inv.shape[0] == 64


def test_attn_temp_scale1_identity():
    assert abs(attn_temperature(1.0) - 1.0) < 1e-6


def test_attn_temp_scale4():
    exp = math.sqrt(0.1 * math.log(4.0) + 1.0)
    assert abs(attn_temperature(4.0) - exp) < 1e-6


def test_curriculum_monotonic():
    assert curriculum_max_len(0) <= curriculum_max_len(100)
    assert curriculum_max_len(100) <= curriculum_max_len(300)
    assert curriculum_max_len(300) <= curriculum_max_len(500)


def test_curriculum_final_32k():
    assert curriculum_max_len(400) == 32768
