"""Capstone — Phi-tiny 270M 训练脚本.

dry-run 默认: 不下载数据 / 不真训, 只验流程.
--train 真跑.
"""
from __future__ import annotations

import argparse
import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainCfg:
    seq_len: int = 1024
    micro_batch: int = 16
    grad_accum: int = 8
    max_step: int = 4000
    base_lr: float = 6e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_pct: float = 0.05
    decay_pct: float = 0.2
    log_every: int = 100
    ckpt_every: int = 1000
    eval_every: int = 500
    seed: int = 42


def wsd(step, max_step, base_lr, w=0.05, d=0.2):
    warmup = int(max_step * w)
    decay_start = int(max_step * (1 - d))
    if step < warmup:
        return base_lr * step / warmup
    if step < decay_start:
        return base_lr
    return base_lr * (1 - (step - decay_start) / (max_step - decay_start))


def sanity_check(model, cfg, device):
    print("\n=== Sanity check ===")
    x = torch.randint(0, model.cfg.vocab_size,
                      (2, 64), device=device)
    y = torch.randint(0, model.cfg.vocab_size,
                      (2, 64), device=device)
    logits = model(x)
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
    assert torch.isfinite(loss), "loss is NaN!"
    print(f"  forward OK, loss = {loss.item():.4f}")

    loss.backward()
    total = sum(p.grad.norm().item() ** 2
                for p in model.parameters() if p.grad is not None) ** 0.5
    print(f"  grad_norm = {total:.4f}")
    assert math.isfinite(total)
    model.zero_grad()


def mock_data_loader(vocab_size, seq_len, batch, rng):
    """无真实数据, 用 random ints 模拟."""
    while True:
        x = torch.from_numpy(rng.integers(0, vocab_size,
                                            (batch, seq_len)).astype(np.int64))
        y = x.clone()  # 教学占位
        yield x, y


def train(cfg: TrainCfg, dry_run: bool = True):
    from phi_tiny_model import PhiTinyConfig, PhiTiny

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[capstone] device={device} dry_run={dry_run}")

    model_cfg = PhiTinyConfig(seq_len=cfg.seq_len)
    model = PhiTiny(model_cfg)
    n_param = sum(p.numel() for p in model.parameters())
    n_no_embed = n_param - model.embed.weight.numel()
    print(f"[capstone] params = {n_param/1e6:.1f}M "
          f"(excl tied embed {n_no_embed/1e6:.1f}M)")

    if dry_run:
        sanity_check(model, cfg, "cpu")
        print(f"\n[capstone] dry-run: WSD lr @ step 0/200/3500/4000:")
        for s in [0, 200, 3500, 4000]:
            print(f"  step {s}: lr = {wsd(s, cfg.max_step, cfg.base_lr):.2e}")
        print("\n[capstone] To actually train: --train")
        return

    model = model.to(device).bfloat16()

    rng = np.random.default_rng(cfg.seed)
    loader = mock_data_loader(model_cfg.vocab_size, cfg.seq_len,
                                cfg.micro_batch, rng)

    opt = torch.optim.AdamW(model.parameters(),
                             lr=cfg.base_lr, betas=(0.9, 0.95),
                             eps=1e-8, weight_decay=cfg.weight_decay)

    print(f"\n[capstone] starting training for {cfg.max_step} steps")
    for step in range(cfg.max_step):
        opt.zero_grad()
        accum_loss = 0.0
        for _ in range(cfg.grad_accum):
            x, y = next(loader)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.flatten(0, 1).float(),
                y.flatten()) / cfg.grad_accum
            loss.backward()
            accum_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                        cfg.grad_clip)
        lr = wsd(step, cfg.max_step, cfg.base_lr,
                  cfg.warmup_pct, cfg.decay_pct)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()

        if step % cfg.log_every == 0:
            print(f"step {step:>5d} loss {accum_loss:.4f} "
                  f"lr {lr:.2e}")

        if step > 0 and step % cfg.ckpt_every == 0:
            ckpt_path = f"ckpt_{step}.pt"
            torch.save({"model": model.state_dict(),
                         "step": step}, ckpt_path)
            print(f"  ckpt → {ckpt_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--micro_batch", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_step", type=int, default=4000)
    p.add_argument("--base_lr", type=float, default=6e-4)
    p.add_argument("--train", action="store_true")
    args = p.parse_args()

    cfg = TrainCfg(
        seq_len=args.seq_len,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        max_step=args.max_step,
        base_lr=args.base_lr,
    )
    train(cfg, dry_run=not args.train)


if __name__ == "__main__":
    main()
