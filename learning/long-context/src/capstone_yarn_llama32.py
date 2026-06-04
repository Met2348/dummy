"""Capstone — Llama-3.2-1B + YaRN scale=4 + LoRA 扩到 32k.

教学骨架: 实际训练需 5090 24G + 5h.
此脚本默认 dry-run, 用 --train 真跑.
"""
from __future__ import annotations

import math
import argparse
from pathlib import Path


def yarn_inv_freq(dim: int, base: float = 10000.0, scale: float = 4.0):
    import torch
    new_base = base * scale ** (dim / (dim - 2))
    inv_freq = 1.0 / (new_base ** (torch.arange(0, dim, 2).float() / dim))
    return inv_freq


def attn_temperature(scale: float = 4.0) -> float:
    return math.sqrt(0.1 * math.log(scale) + 1.0)


def inject_yarn(model, scale: float = 4.0, new_max_pos: int = 32768):
    """对 Llama Attention 的 rotary_emb 注入 YaRN."""
    import torch
    cfg = model.config
    cfg.max_position_embeddings = new_max_pos
    cfg.rope_scaling = {"type": "yarn", "factor": scale}

    head_dim = cfg.hidden_size // cfg.num_attention_heads
    new_inv = yarn_inv_freq(head_dim, scale=scale)

    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, "rotary_emb"):
            attn.rotary_emb.inv_freq.data = new_inv.to(
                attn.rotary_emb.inv_freq.device)

    print(f"[yarn] scale={scale}, new_max_pos={new_max_pos}, "
          f"attn_temp={attn_temperature(scale):.4f}")
    return model


def setup_lora(model):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)


def curriculum_max_len(step: int) -> int:
    if step < 100:
        return 8192
    if step < 300:
        return 16384
    return 32768


def main(args):
    print(f"[capstone] dry-run={not args.train}")
    print(f"[capstone] model={args.model}")
    print(f"[capstone] scale={args.scale}, target_ctx={args.target_ctx}")
    print(f"[capstone] steps={args.steps}, lr={args.lr}")
    print(f"[capstone] curriculum:")
    for s in [0, 50, 150, 350]:
        print(f"  step {s:>4d}: max_len = {curriculum_max_len(s)}")

    if not args.train:
        print("\n[capstone] dry-run, no training.")
        print("[capstone] to actually train: --train --steps 500")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("[capstone] loading model …")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("[capstone] injecting YaRN …")
    model = inject_yarn(model, scale=args.scale, new_max_pos=args.target_ctx)

    print("[capstone] adding LoRA …")
    model = setup_lora(model)
    model.print_trainable_parameters()

    print("[capstone] (省略 data loader + Trainer 实例化)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--scale", type=float, default=4.0)
    p.add_argument("--target_ctx", type=int, default=32768)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--train", action="store_true")
    args = p.parse_args()
    main(args)
