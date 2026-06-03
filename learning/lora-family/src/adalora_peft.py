"""AdaLoRA peft 调包版。

对应 minimal: adalora_minimal.py
对应 lecture: lectures/02-adalora.md（第 21 张幻灯片）

peft 的命名 (lora_A, lora_E, lora_B) 对应论文 (Q^T, Λ, P)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from peft import AdaLoraConfig, TaskType, get_peft_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent))
from common import print_param_summary  # noqa: E402


def build_peft_model(
    r_init: int = 12,
    r_final: int = 8,
    alpha: int = 16,
    total_step: int = 1000,
    target_modules: list | None = None,
) -> torch.nn.Module:
    """构造 peft AdaLoRA 模型。

    Notes:
        peft AdaLoraConfig 要求 total_step 显式给定（用于内部调度）。
    """
    if target_modules is None:
        target_modules = ["c_attn"]
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    # peft AdaLoRA schedule 要求: total_step - tinit - tfinal > 0
    # 我们选 tinit=10%, tfinal=60%（剩 30% 给 fine-tune 收尾）
    tinit = int(total_step * 0.1)
    tfinal = int(total_step * 0.6)
    config = AdaLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        init_r=r_init,
        target_r=r_final,
        beta1=0.85,
        beta2=0.85,
        tinit=tinit,
        tfinal=tfinal,
        deltaT=10,
        lora_alpha=alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
        orth_reg_weight=0.5,
        total_step=total_step,
    )
    return get_peft_model(base, config)


def main() -> None:
    torch.manual_seed(42)
    model = build_peft_model(r_init=12, r_final=8, alpha=16)
    print_param_summary(model, "peft AdaLoRA (r_init=12, r_final=8)")

    print("\npeft 内部 layer 0 参数：")
    for name, p in model.named_parameters():
        if "h.0.attn.c_attn" in name and ("lora_A" in name or "lora_B" in name or "lora_E" in name):
            print(f"  {name}: shape={tuple(p.shape)}, trainable={p.requires_grad}")

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    enc = tok("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    print(f"\n前向输出 logits.shape={tuple(out.logits.shape)}")


if __name__ == "__main__":
    main()
