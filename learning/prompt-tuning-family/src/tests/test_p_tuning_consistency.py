"""
P-Tuning v1 一致性测试（弱一致）。

由于 LSTM 内部参数初始化对 seed 敏感且 peft 与 minimal 的初始化顺序不同，
不强求 logits 数值一致；验证：
  1) 前向 logits 形状一致
  2) 可训练参数量同量级
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from common import count_parameters  # noqa: E402
from p_tuning_minimal import PTuningGPT2  # noqa: E402
from p_tuning_peft import build_peft_model  # noqa: E402


def test_shape_and_param_order() -> None:
    torch.manual_seed(42)
    P = 10

    m_min = PTuningGPT2(prompt_length=P, encoder_hidden=256).eval()
    m_peft = build_peft_model(prompt_length=P, hidden=256).eval()

    tok = m_min.tokenizer
    enc = tok("hello world", return_tensors="pt", padding=True)

    with torch.no_grad():
        out_min = m_min(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        out_peft = m_peft(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
        )

    print(f"minimal logits: {out_min.logits.shape}")
    print(f"peft    logits: {out_peft.logits.shape}")
    assert out_min.logits.shape == out_peft.logits.shape, (
        f"Shape 不一致: minimal={out_min.logits.shape}, peft={out_peft.logits.shape}"
    )

    n_min, _ = count_parameters(m_min)
    n_peft, _ = count_parameters(m_peft)
    ratio = max(n_min, n_peft) / min(n_min, n_peft)
    print(f"\nminimal trainable: {n_min:,}")
    print(f"peft    trainable: {n_peft:,}")
    print(f"ratio: {ratio:.4f}x")

    assert ratio < 2.0, f"参数量差异过大: {ratio}x"
    print("\n[PASS] 形状一致 + 参数量同量级")


if __name__ == "__main__":
    test_shape_and_param_order()
