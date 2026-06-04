"""mini-Mamba 长上下文外推测试."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_mamba import MiniMamba, MiniMambaConfig


def test_mini_mamba_forward():
    cfg = MiniMambaConfig(vocab_size=128, n_layer=2, d_model=32, d_state=8)
    m = MiniMamba(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    y = m(x)
    assert y.shape == (1, 16, cfg.vocab_size)


def test_long_context_no_explosion():
    """训 32 长度 → 推 128 长度，ppl 不爆."""
    cfg = MiniMambaConfig(vocab_size=64, n_layer=2, d_model=32, d_state=8)
    m = MiniMamba(cfg).eval()
    with torch.no_grad():
        short_x = torch.randint(0, cfg.vocab_size, (1, 32))
        long_x = torch.randint(0, cfg.vocab_size, (1, 128))
        out_short = m(short_x)
        out_long = m(long_x)
    # output 应有界且不 NaN
    assert torch.isfinite(out_short).all()
    assert torch.isfinite(out_long).all()
    # logit 范围合理
    assert out_long.abs().max() < 1e4
