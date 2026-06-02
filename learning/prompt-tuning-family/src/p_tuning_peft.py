"""
P-Tuning v1 peft 调包版。

对应 minimal: p_tuning_minimal.py
对应 lecture: lectures/03-p-tuning.md（第 27 张幻灯片）

peft 中 P-Tuning v1 用 PromptEncoderConfig（LSTM 模式），
内部 prompt_encoder.default 结构对应 minimal 的 PromptEncoder。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from peft import (
    PromptEncoderConfig,
    PromptEncoderReparameterizationType,
    TaskType,
    get_peft_model,
)
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent))
from common import print_param_summary  # noqa: E402


def build_peft_model(
    prompt_length: int = 10,
    hidden: int = 256,
) -> torch.nn.Module:
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    config = PromptEncoderConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=prompt_length,
        encoder_reparameterization_type=PromptEncoderReparameterizationType.LSTM,
        encoder_hidden_size=hidden,
        encoder_num_layers=2,
    )
    return get_peft_model(base, config)


def main() -> None:
    torch.manual_seed(42)
    model = build_peft_model(prompt_length=10)
    print_param_summary(model, "peft P-Tuning v1(p=10, h=256)")

    print("\npeft 内部可训练参数：")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {name}: {tuple(p.shape)} = {p.numel():,}")

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    enc = tok("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    print(f"\n前向输出 logits.shape={tuple(out.logits.shape)}")


if __name__ == "__main__":
    main()
