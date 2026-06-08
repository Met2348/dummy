"""Capstone - train mini-Mamba."""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from mini_mamba import MiniMamba, MiniMambaConfig


def mock_batch(vocab, b, t, device):
    x = torch.randint(0, vocab, (b, t), device=device)
    y = torch.cat([x[:, 1:], torch.zeros_like(x[:, :1])], dim=1)
    return x, y


def train(steps: int = 50, device: str = "cpu"):
    cfg = MiniMambaConfig(vocab_size=256, n_layer=2, d_model=64, d_state=8)
    m = MiniMamba(cfg).to(device).train()
    n_p = sum(p.numel() for p in m.parameters())
    print(f"params: {n_p:,}")
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    losses = []
    t0 = time.time()
    for s in range(1, steps + 1):
        x, y = mock_batch(cfg.vocab_size, 4, 32, device)
        logits = m(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        losses.append(loss.item())
        if s in {1, steps // 4, steps // 2, steps}:
            print(f"  step {s:4d}  loss {loss.item():.4f}")
    dt = time.time() - t0
    print(f"\nfinal loss {losses[-1]:.4f}  ppl {math.exp(min(losses[-1], 20)):.2f}  ({dt:.1f}s)")
    return losses


if __name__ == "__main__":
    train(steps=80)
