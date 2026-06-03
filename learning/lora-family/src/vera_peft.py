"""VeRA peft 调包版。

peft 0.10+ 支持 VeraConfig。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from peft import VeraConfig, TaskType, get_peft_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent))
from common import print_param_summary  # noqa: E402


def build_peft_model(
    r: int = 256,
    target_modules: list | None = None,
    d_initial: float = 0.1,
) -> torch.nn.Module:
    """构造 peft VeRA 模型。"""
    if target_modules is None:
        target_modules = ["c_attn"]
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    config = VeraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        target_modules=target_modules,
        vera_dropout=0.0,
        d_initial=d_initial,
        save_projection=True,  # 共享 A, B 也存储（True 才能完整复现）
        projection_prng_key=42,
    )
    return get_peft_model(base, config)


def main() -> None:
    torch.manual_seed(42)
    model = build_peft_model(r=256, d_initial=0.1)
    print_param_summary(model, "peft VeRA (r=256)")

    print("\npeft 内部 layer 0 可训练参数：")
    for name, p in model.named_parameters():
        if "h.0.attn.c_attn.vera" in name:
            print(f"  {name}: shape={tuple(p.shape)}, trainable={p.requires_grad}")

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    enc = tok("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    print(f"\n前向输出 logits.shape={tuple(out.logits.shape)}")


if __name__ == "__main__":
    main()
