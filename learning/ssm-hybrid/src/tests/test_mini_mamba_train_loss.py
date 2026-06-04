"""mini-Mamba 训练 loss 下降验证."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_mamba import MiniMamba, MiniMambaConfig


def test_loss_decreases_over_50_steps():
    torch.manual_seed(0)
    cfg = MiniMambaConfig(vocab_size=64, n_layer=2, d_model=32, d_state=8)
    m = MiniMamba(cfg).train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    losses = []
    for _ in range(50):
        x = torch.randint(0, cfg.vocab_size, (4, 8))
        y = torch.cat([x[:, 1:], torch.zeros_like(x[:, :1])], dim=1)
        logits = m(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        loss.backward()
        opt.step(); opt.zero_grad()
        losses.append(loss.item())
    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10
