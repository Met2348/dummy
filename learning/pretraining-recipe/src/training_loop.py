"""标准 pretraining 训练 loop."""
from __future__ import annotations

import math
import time
import torch
import torch.nn.functional as F
from torch.amp import autocast


def split_param_groups(model, weight_decay=0.1):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2 and "norm" not in n.lower():
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def make_optimizer(model, lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1):
    return torch.optim.AdamW(
        split_param_groups(model, weight_decay),
        lr=lr, betas=betas, eps=1e-8,
    )


def wsd_lr(step, max_step, base_lr, warmup_pct=0.05, decay_pct=0.2):
    warmup = int(max_step * warmup_pct)
    decay_start = int(max_step * (1 - decay_pct))
    if step < warmup:
        return base_lr * step / warmup
    if step < decay_start:
        return base_lr
    return base_lr * (1 - (step - decay_start) / (max_step - decay_start))


def set_lr(opt, lr):
    for g in opt.param_groups:
        g["lr"] = lr


class EmaTracker:
    def __init__(self, alpha=0.99):
        self.alpha = alpha
        self.ema = None

    def update(self, x):
        self.ema = x if self.ema is None \
            else self.alpha * self.ema + (1 - self.alpha) * x

    def is_spike(self, x, threshold=3.0):
        return self.ema is not None and x > threshold * self.ema


def train_step(model, x, y, opt, grad_clip=1.0,
                grad_accum=1, step_idx=0):
    """单 (logical) step, 内含 micro_batch / grad_accum."""
    opt.zero_grad()
    micro_loss = 0.0
    for _ in range(grad_accum):
        with autocast(device_type="cuda" if torch.cuda.is_available()
                       else "cpu", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(
                logits.flatten(0, 1), y.flatten()) / grad_accum
        loss.backward()
        micro_loss += loss.item()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step()
    return micro_loss, gn.item()


def train(model, data_loader, max_step=1000, base_lr=6e-4,
          grad_accum=1, log_every=100, ckpt_every=500,
          ckpt_path="ckpt.pt"):
    opt = make_optimizer(model, lr=base_lr)
    ema = EmaTracker()
    t0 = time.time()

    for step in range(max_step):
        x, y = next(data_loader)
        x, y = x.to(model.device if hasattr(model, "device") else "cpu"), \
               y.to(model.device if hasattr(model, "device") else "cpu")
        lr = wsd_lr(step, max_step, base_lr)
        set_lr(opt, lr)
        loss, gn = train_step(model, x, y, opt, grad_accum=grad_accum,
                                step_idx=step)
        ema.update(loss)

        if ema.is_spike(loss):
            print(f"[SPIKE] step {step} loss {loss:.3f} ema {ema.ema:.3f}")
            continue

        if step % log_every == 0:
            tok = x.shape[0] * x.shape[1] * grad_accum
            dt = time.time() - t0
            tok_s = tok / dt if dt > 0 else 0
            print(f"step {step:>5d} loss {loss:.4f} ema {ema.ema:.4f} "
                  f"gn {gn:.2f} lr {lr:.2e} tok/s {tok_s:.0f}")
            t0 = time.time()

        if step > 0 and step % ckpt_every == 0:
            torch.save({"model": model.state_dict(),
                         "step": step}, ckpt_path)
            print(f"  ckpt saved @ step {step}")


if __name__ == "__main__":
    print("Training loop interface ready. See capstone_train.py for actual run.")
