"""(IA)³ — peft 库调包版（三轨第三版）。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from peft import IA3Config, TaskType, get_peft_model
from transformers import GPT2LMHeadModel

sys.path.append(str(Path(__file__).parent))
from common import print_param_summary  # noqa: E402


def build_ia3_model_peft(target_modules=None, feedforward_modules=None):
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    if target_modules is None:
        target_modules = ["c_attn", "c_proj", "c_fc"]
    if feedforward_modules is None:
        feedforward_modules = ["c_fc"]
    config = IA3Config(
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        feedforward_modules=feedforward_modules,
    )
    return get_peft_model(base, config)


def main() -> None:
    torch.manual_seed(42)
    model = build_ia3_model_peft()
    print_param_summary(model, "peft (IA)^3")

    print("\npeft 内部 layer 0 可训练参数:")
    cnt = 0
    for name, p in model.named_parameters():
        if p.requires_grad and "h.0" in name and cnt < 5:
            print(f"  {name}: shape={tuple(p.shape)}")
            cnt += 1


if __name__ == "__main__":
    main()
