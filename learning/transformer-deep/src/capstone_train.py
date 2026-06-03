"""Capstone — train GPT-mini (smoke 模式 / 真训模式).

教学目标：
    1. 完整训练循环 (AdamW + cosine lr + grad clip)
    2. forward / backward / step / log
    3. smoke: 100 step + mock data
    4. real: 真实 jsonl + SP tokenizer

运行：
    python capstone_train.py --steps 100   # smoke
    python capstone_train.py --real --jsonl ... --tokenizer ...
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpt_mini import GPTMini, GPTMiniConfig


def lr_at_step(step: int, warmup: int, max_steps: int,
               lr_max: float = 3e-4, lr_min: float = 3e-5) -> float:
    if step < warmup:
        return lr_max * step / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def mock_batch(vocab_size: int, batch: int, seq: int, device):
    x = torch.randint(0, vocab_size, (batch, seq), device=device)
    y = torch.cat([x[:, 1:], torch.zeros_like(x[:, :1])], dim=1)
    return x, y


def train(steps: int = 100, device: str | None = None,
          cfg: GPTMiniConfig | None = None) -> dict:
    cfg = cfg or GPTMiniConfig(vocab_size=1024, n_layer=4, n_head=8,
                                n_kv=2, d_model=256, d_ff=512, max_seq=128)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTMini(cfg).to(device)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_p:,}  device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95),
                            weight_decay=0.1)
    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
        lr = lr_at_step(step, warmup=max(10, steps // 10), max_steps=steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = mock_batch(cfg.vocab_size, batch=8, seq=64, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
        if step == 1 or step % max(1, steps // 10) == 0:
            print(f"  step {step:5d}  loss {loss.item():.4f}  lr {lr:.2e}")
    dt = time.time() - t0
    final_ppl = math.exp(min(losses[-1], 20))
    print(f"\nDone {steps} steps in {dt:.1f}s  final loss {losses[-1]:.3f}  ppl {final_ppl:.2f}")
    return {"losses": losses, "ppl": final_ppl, "n_params": n_p}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    train(steps=args.steps, device=args.device)


if __name__ == "__main__":
    main()
