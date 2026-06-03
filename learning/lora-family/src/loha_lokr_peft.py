"""LoHa + LoKr peft 调包版。

peft 0.10+ 通过 LoHaConfig, LoKrConfig 支持。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from peft import LoHaConfig, LoKrConfig, TaskType, get_peft_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent))
from common import print_param_summary  # noqa: E402


def build_peft_loha(r: int = 8, alpha: int = 16, target_modules: list | None = None):
    if target_modules is None:
        target_modules = ["c_attn"]
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    config = LoHaConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        alpha=alpha,
        target_modules=target_modules,
        rank_dropout=0.0,
        module_dropout=0.0,
    )
    return get_peft_model(base, config)


def build_peft_lokr(factor: int = 8, r: int = 4, alpha: int = 4, target_modules: list | None = None):
    if target_modules is None:
        target_modules = ["c_attn"]
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    config = LoKrConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        alpha=alpha,
        target_modules=target_modules,
        decompose_both=False,
        decompose_factor=factor,
    )
    return get_peft_model(base, config)


def main() -> None:
    print("=" * 60)
    print("peft LoHa / LoKr 在 GPT-2 上的兼容性说明")
    print("=" * 60)
    print("peft 的 LoHaConfig / LoKrConfig 当前**不支持**")
    print("GPT-2 用的 transformers.pytorch_utils.Conv1D")
    print("（仅支持 nn.Linear, nn.Conv1d, nn.Conv2d）")
    print()
    print("在 LLaMA、BERT 等用 nn.Linear 的模型上可直接调用：")
    print('  LoHaConfig(r=8, alpha=16, target_modules=["q_proj", "v_proj"])')
    print('  LoKrConfig(r=4, alpha=4,  target_modules=["q_proj", "v_proj"])')
    print()
    print("=" * 60)
    print("尝试在 GPT-2 上构造（预期 ValueError）")
    print("=" * 60)
    try:
        torch.manual_seed(42)
        loha_model = build_peft_loha(r=8, alpha=16)
        print_param_summary(loha_model, "peft LoHa (r=8)")
    except ValueError as e:
        print(f"[预期错误] {type(e).__name__}: {str(e)[:120]}")
    try:
        torch.manual_seed(42)
        lokr_model = build_peft_lokr(factor=32, r=4)
        print_param_summary(lokr_model, "peft LoKr")
    except ValueError as e:
        print(f"[预期错误] {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
