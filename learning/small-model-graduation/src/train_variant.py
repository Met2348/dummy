"""5 个 ckpt 训练统一入口.

dry-run 默认: 不真训. --train 真跑.
"""
from __future__ import annotations

import argparse
import os
import sys
import math
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


@dataclass
class VariantCfg:
    variant: str
    model_name: str
    n_param: int
    seq_len: int
    micro_batch: int
    grad_accum: int
    max_step: int
    base_lr: float
    data_desc: str
    target_loss: float
    notes: str = ""


VARIANTS = {
    "A": VariantCfg(
        variant="A", model_name="vanilla_gpt2_124m",
        n_param=124_000_000,
        seq_len=512, micro_batch=32, grad_accum=4,
        max_step=3000, base_lr=6e-4,
        data_desc="TinyStories + OpenWebText",
        target_loss=3.5,
        notes="Vanilla GPT-2 baseline",
    ),
    "B": VariantCfg(
        variant="B", model_name="vanilla_gpt2_124m",
        n_param=124_000_000,
        seq_len=512, micro_batch=32, grad_accum=4,
        max_step=3000, base_lr=6e-4,
        data_desc="Cosmopedia + filtered web (HIGH QUALITY)",
        target_loss=3.2,
        notes="改数据, 模型/训练同 A",
    ),
    "C": VariantCfg(
        variant="C", model_name="phi_tiny_270m",
        n_param=270_000_000,
        seq_len=1024, micro_batch=16, grad_accum=8,
        max_step=4000, base_lr=6e-4,
        data_desc="Cosmopedia + filtered web",
        target_loss=2.9,
        notes="改架构, Phi-tiny GQA+SwiGLU+RoPE+RMSNorm",
    ),
    "D": VariantCfg(
        variant="D", model_name="phi_tiny_270m_yarn",
        n_param=270_000_000,
        seq_len=8192, micro_batch=2, grad_accum=8,
        max_step=100, base_lr=5e-5,
        data_desc="FineWeb-Edu long doc",
        target_loss=2.9,
        notes="从 C ckpt resume, YaRN scale=4, LoRA r=16, 100 step",
    ),
    "E": VariantCfg(
        variant="E", model_name="phi_tiny_270m_curriculum",
        n_param=270_000_000,
        seq_len=1024, micro_batch=16, grad_accum=8,
        max_step=4000, base_lr=6e-4,
        data_desc="curriculum: web → code → math → long doc",
        target_loss=2.8,
        notes="一气训长 ctx + 课程",
    ),
}


def load_model(name: str):
    """根据 variant 选择 model class."""
    if name.startswith("vanilla"):
        from vanilla_gpt2 import VanillaGPT2, GPT2Config
        return VanillaGPT2(GPT2Config())
    elif name.startswith("phi_tiny"):
        sys.path.insert(0, os.path.join(
            os.path.dirname(__file__), "../../pretraining-recipe/src"))
        from phi_tiny_model import PhiTinyConfig, PhiTiny
        return PhiTiny(PhiTinyConfig())
    else:
        raise ValueError(name)


def wsd_lr(step, max_step, base_lr, w=0.05, d=0.2):
    warmup = int(max_step * w)
    decay_start = int(max_step * (1 - d))
    if step < warmup:
        return base_lr * step / warmup
    if step < decay_start:
        return base_lr
    return base_lr * (1 - (step - decay_start) / (max_step - decay_start))


def mock_loader(vocab, seq, batch, rng):
    while True:
        x = torch.from_numpy(
            rng.integers(0, vocab, (batch, seq)).astype(np.int64))
        y = x.clone()
        yield x, y


def train(cfg: VariantCfg, dry: bool = True):
    print(f"\n=== Variant {cfg.variant} ===")
    print(f"  model:    {cfg.model_name} ({cfg.n_param/1e6:.0f}M)")
    print(f"  data:     {cfg.data_desc}")
    print(f"  training: seq={cfg.seq_len} batch="
          f"{cfg.micro_batch}×{cfg.grad_accum}={cfg.micro_batch*cfg.grad_accum} "
          f"step={cfg.max_step} lr={cfg.base_lr}")
    print(f"  target loss: {cfg.target_loss}")
    print(f"  notes: {cfg.notes}")

    if dry:
        print(f"\n[dry-run] WSD lr @ key steps:")
        for s in [0, cfg.max_step // 4, cfg.max_step // 2,
                   cfg.max_step * 4 // 5, cfg.max_step]:
            print(f"    step {s}: lr={wsd_lr(s, cfg.max_step, cfg.base_lr):.2e}")
        print(f"\n[dry-run] use --train to actually run")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  device={device}")
    model = load_model(cfg.model_name).to(device).bfloat16()
    vocab = model.cfg.vocab_size if hasattr(model, "cfg") \
        else model.tok_embed.num_embeddings
    rng = np.random.default_rng(42)
    loader = mock_loader(vocab, cfg.seq_len, cfg.micro_batch, rng)

    opt = torch.optim.AdamW(model.parameters(),
                             lr=cfg.base_lr, betas=(0.9, 0.95),
                             eps=1e-8, weight_decay=0.1)

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = wsd_lr(step, cfg.max_step, cfg.base_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()
        if step % 100 == 0:
            print(f"  step {step:>5d} loss {accum_loss:.4f} lr {lr:.2e}")

    ckpt = f"ckpt_{cfg.variant}.pt"
    torch.save({"model": model.state_dict(), "variant": cfg.variant}, ckpt)
    print(f"\n  saved → {ckpt}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=list(VARIANTS.keys()), required=True)
    p.add_argument("--train", action="store_true")
    args = p.parse_args()
    cfg = VARIANTS[args.variant]
    train(cfg, dry=not args.train)


if __name__ == "__main__":
    main()
