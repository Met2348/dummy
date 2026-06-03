"""GPT-mini forward / backward smoke."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpt_mini import GPTMini, GPTMiniConfig


def _tiny_cfg():
    return GPTMiniConfig(vocab_size=128, n_layer=2, n_head=4, n_kv=2,
                         d_model=64, d_ff=128, max_seq=32)


def test_forward_shape():
    cfg = _tiny_cfg()
    m = GPTMini(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = m(x)
    assert y.shape == (2, 16, cfg.vocab_size)


def test_backward_flows():
    cfg = _tiny_cfg()
    m = GPTMini(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = m(x).sum()
    y.backward()
    for n, p in m.named_parameters():
        # tied lm_head 与 tok_embed 共享，只检查 embed
        if p.requires_grad and p.grad is None:
            raise AssertionError(f"no grad: {n}")


def test_no_nan():
    cfg = _tiny_cfg()
    m = GPTMini(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    y = m(x)
    assert torch.isfinite(y).all()


def test_param_count_reasonable():
    """tiny cfg 参数应在合理范围."""
    cfg = _tiny_cfg()
    m = GPTMini(cfg)
    n = sum(p.numel() for p in m.parameters())
    # vocab 128 × d 64 = 8k for embed
    # 2 block × ~30k = 60k
    assert 5_000 < n < 100_000, f"params {n}"
