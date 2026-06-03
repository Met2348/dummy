"""PiSSA peft 调包版。

LoraConfig(init_lora_weights="pissa") 在 peft 0.12+ 支持。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent))
from common import print_param_summary  # noqa: E402


def build_peft_model(
    r: int = 8,
    alpha: int | None = None,
    target_modules: list | None = None,
    init_method: str = "pissa",
) -> torch.nn.Module:
    """构造 peft PiSSA / OLoRA 模型。

    init_method: "pissa"（SVD top-r）或 "olora"（QR top-r）
    """
    if target_modules is None:
        target_modules = ["c_attn"]
    if alpha is None:
        alpha = r
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        init_lora_weights=init_method,
    )
    return get_peft_model(base, config)


def main() -> None:
    torch.manual_seed(42)
    model = build_peft_model(r=8, alpha=8, init_method="pissa")
    print_param_summary(model, "peft PiSSA (r=8, init=pissa)")

    print("\npeft 内部 layer 0 参数：")
    for name, p in model.named_parameters():
        if "h.0.attn.c_attn.lora" in name:
            print(f"  {name}: shape={tuple(p.shape)}")

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    enc = tok("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    print(f"\n前向输出 logits.shape={tuple(out.logits.shape)}")


if __name__ == "__main__":
    main()
