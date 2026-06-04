"""Capstone — mini-MoE 训练对照 dense baseline."""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from mini_moe import MiniMoE, MiniMoEConfig


def mock_batch(vocab, b, t, device):
    x = torch.randint(0, vocab, (b, t), device=device)
    y = torch.cat([x[:, 1:], torch.zeros_like(x[:, :1])], dim=1)
    return x, y


def train(model, steps=100, lr=3e-4, vocab=256, device="cpu"):
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    losses = []
    for s in range(1, steps + 1):
        x, y = mock_batch(vocab, 8, 32, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        losses.append(loss.item())
        if s in {1, steps // 4, steps // 2, steps}:
            print(f"  step {s:4d}  loss {loss.item():.4f}")
    return losses


def main(steps: int = 100):
    cfg = MiniMoEConfig(vocab_size=256, n_layer=2, d_model=64,
                        n_head=4, n_routed=4, top_k=2)
    print(f"\n=== mini-MoE (n_routed={cfg.n_routed}, top_k={cfg.top_k}) ===")
    moe = MiniMoE(cfg)
    n_p = sum(p.numel() for p in moe.parameters())
    print(f"params: {n_p:,}")
    t0 = time.time()
    moe_losses = train(moe, steps=steps, vocab=cfg.vocab_size)
    dt = time.time() - t0
    print(f"final loss {moe_losses[-1]:.4f}  ppl {math.exp(min(moe_losses[-1], 20)):.2f}  ({dt:.1f}s)")
    return moe_losses


if __name__ == "__main__":
    main(steps=100)
