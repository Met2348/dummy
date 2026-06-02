"""
Prompt Tuning peft 调包版（用于与 minimal 对照）。

对应 minimal: prompt_tuning_minimal.py
对应 lecture: lectures/02-prompt-tuning.md（第 26 张幻灯片）

peft 的 PromptTuningConfig 内部等价于：
  - 一个 nn.Embedding(p, d) 作为 prompt 来源
  - 把 prompt 拼到 input embedding 前
  - 冻结 LM、仅训练 embedding
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent))
from common import print_param_summary  # noqa: E402


def build_peft_model(
    prompt_length: int = 10,
    init_type: PromptTuningInit = PromptTuningInit.RANDOM,
) -> torch.nn.Module:
    """构造 peft PromptTuning 模型。

    内部存储：model.prompt_encoder.default.embedding.weight ∈ R^(p, d)
    """
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=init_type,
        num_virtual_tokens=prompt_length,
        tokenizer_name_or_path="gpt2",
    )
    return get_peft_model(base, config)


def main() -> None:
    torch.manual_seed(42)
    model = build_peft_model(prompt_length=10)
    print_param_summary(model, "peft PromptTuning(prompt_length=10)")
    # Expected: trainable=7,680（与 minimal 一致）

    # 打印 peft 内部存储结构
    print("\npeft 内部可训练参数：")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {name}: shape={tuple(p.shape)}")

    # 前向
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    enc = tok("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    print(f"\n前向输出 logits.shape={tuple(out.logits.shape)}")


if __name__ == "__main__":
    main()
