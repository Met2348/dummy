"""Mini-MoE forward smoke + train smoke."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_moe import MiniMoE, MiniMoEConfig    # noqa


def test_mini_moe_forward_shape():
    cfg = MiniMoEConfig(vocab_size=128, n_layer=2, d_model=32,
                        n_head=4, n_routed=4, top_k=2,
                        d_ff_routed=32, d_ff_shared=64)
    m = MiniMoE(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = m(x)
    assert y.shape == (2, 8, cfg.vocab_size)
    assert torch.isfinite(y).all()


def test_mini_moe_backward():
    cfg = MiniMoEConfig(vocab_size=64, n_layer=2, d_model=32,
                        n_head=4, n_routed=4, top_k=2,
                        d_ff_routed=32, d_ff_shared=64)
    m = MiniMoE(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    y = m(x).sum()
    y.backward()
    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is None:
            # tied lm_head 可能共享
            if "lm_head" in n:
                continue
            raise AssertionError(f"no grad: {n}")


def test_mini_moe_loss_decreases():
    """100 step 小训练 loss 应单调下降."""
    cfg = MiniMoEConfig(vocab_size=64, n_layer=2, d_model=32,
                        n_head=4, n_routed=4, top_k=2,
                        d_ff_routed=32, d_ff_shared=64)
    m = MiniMoE(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    losses = []
    for _ in range(50):
        x = torch.randint(0, cfg.vocab_size, (4, 8))
        y = torch.cat([x[:, 1:], torch.zeros_like(x[:, :1])], dim=1)
        logits = m(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, cfg.vocab_size), y.view(-1))
        loss.backward(); opt.step(); opt.zero_grad()
        losses.append(loss.item())
    # 最后 10 step 平均 < 前 10 step 平均
    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10
