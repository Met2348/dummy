"""
Prefix Tuning peft 调包版。

对应 minimal: prefix_tuning_minimal.py
对应 lecture: lectures/01-prefix-tuning.md（第 26 张幻灯片）

peft 的 PrefixTuningConfig + prefix_projection=True 内部结构：
  embedding (p, d) → Linear(d, hidden) → Tanh → Linear(hidden, L*2*d)
等价于 minimal 的 P_low + reparam MLP。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from peft import PrefixTuningConfig, TaskType, get_peft_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent))
from common import print_param_summary  # noqa: E402


def build_peft_model(
    prefix_length: int = 10,
    mid_dim: int = 512,
    use_reparam: bool = True,
) -> torch.nn.Module:
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=prefix_length,
        encoder_hidden_size=mid_dim,
        prefix_projection=use_reparam,
    )
    return get_peft_model(base, config)


def main() -> None:
    torch.manual_seed(42)
    model = build_peft_model(prefix_length=10)
    print_param_summary(model, "peft PrefixTuning(p=10, mid=512, projection=True)")

    print("\npeft 内部可训练参数：")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {name}: shape={tuple(p.shape)}, numel={p.numel():,}")

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    enc = tok("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    print(f"\n前向输出 logits.shape={tuple(out.logits.shape)}")


if __name__ == "__main__":
    main()
