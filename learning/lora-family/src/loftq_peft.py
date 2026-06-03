"""LoftQ peft 调包版。

peft 0.10+ 通过 LoraConfig(init_lora_weights="loftq", loftq_config=...) 支持。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from peft import LoraConfig, LoftQConfig, TaskType, get_peft_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent))
from common import print_param_summary  # noqa: E402


def build_peft_model(
    r: int = 8,
    alpha: int | None = None,
    n_iter: int = 5,
    target_modules: list | None = None,
):
    """构造 peft LoftQ 模型。"""
    if alpha is None:
        alpha = r
    if target_modules is None:
        target_modules = ["c_attn"]
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    loftq_cfg = LoftQConfig(loftq_bits=4, loftq_iter=n_iter)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        init_lora_weights="loftq",
        loftq_config=loftq_cfg,
    )
    return get_peft_model(base, config)


def main() -> None:
    torch.manual_seed(42)
    print("尝试 peft LoftQ (n_iter=5)...")
    try:
        model = build_peft_model(r=8, alpha=8, n_iter=5)
        print_param_summary(model, "peft LoftQ (r=8, T=5)")
        print()
        for name, p in model.named_parameters():
            if "h.0.attn.c_attn.lora" in name:
                print(f"  {name}: shape={tuple(p.shape)}")
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
