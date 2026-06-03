"""LoRA peft 调包版。

对应 minimal: lora_minimal.py
对应 lecture: lectures/01-lora.md（第 25-26 张幻灯片）

peft 的 LoraConfig 默认 target_modules 在 GPT-2 上需要手动指定为 ["c_attn"]。
内部参数布局：
  base_model.model.transformer.h.<i>.attn.c_attn.lora_A.default.weight  shape=(r, d_in=768)
  base_model.model.transformer.h.<i>.attn.c_attn.lora_B.default.weight  shape=(d_out=2304, r)
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
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: list | None = None,
) -> torch.nn.Module:
    """构造 peft LoRA 模型，配置与 minimal 对齐。"""
    if target_modules is None:
        target_modules = ["c_attn"]
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        init_lora_weights=True,  # 默认 Kaiming(A) + zero(B)
    )
    return get_peft_model(base, config)


def main() -> None:
    torch.manual_seed(42)
    model = build_peft_model(r=8, alpha=16)
    print_param_summary(model, "peft LoRA (r=8, target=c_attn)")

    print("\npeft 内部可训练参数（前 4 个）：")
    count = 0
    for name, p in model.named_parameters():
        if p.requires_grad and count < 4:
            print(f"  {name}")
            print(f"    shape={tuple(p.shape)}, numel={p.numel():,}")
            count += 1
    print(f"  ... (共 {sum(1 for _ in model.named_parameters() if _[1].requires_grad)} 个可训练参数)")

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    enc = tok("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    print(f"\n前向输出 logits.shape={tuple(out.logits.shape)}")


if __name__ == "__main__":
    main()
