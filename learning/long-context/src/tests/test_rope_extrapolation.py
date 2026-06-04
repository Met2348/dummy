"""PI / NTK / YaRN 外推性测试."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rope_pi import pi_cos_sin
from rope_ntk import ntk_cos_sin
from rope_yarn import yarn_cos_sin


def test_pi_shape():
    cos, sin = pi_cos_sin(t=64, dim=32, scale_factor=8)
    assert cos.shape == (64, 16)


def test_ntk_shape():
    cos, sin = ntk_cos_sin(t=64, dim=32, scale_factor=8)
    assert cos.shape == (64, 16)


def test_yarn_shape_and_attn_scale():
    cos, sin, scale = yarn_cos_sin(t=64, dim=32, factor=8, original_max_pos=8)
    assert cos.shape == (64, 16)
    assert 0 < scale <= 1.0


def test_long_context_finite():
    """三种方法在 4× 长 context 下不出 NaN/inf."""
    for fn in [
        lambda: pi_cos_sin(t=128, dim=32, scale_factor=4),
        lambda: ntk_cos_sin(t=128, dim=32, scale_factor=4),
        lambda: yarn_cos_sin(t=128, dim=32, factor=4, original_max_pos=32)[:2],
    ]:
        cos, sin = fn()
        assert torch.isfinite(cos).all()
        assert torch.isfinite(sin).all()
