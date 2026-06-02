"""
Prefix Tuning 一致性测试（弱一致）。

由于 minimal 与 peft 的 reparameterization MLP 内部初始化细节可能不同
（即使结构一致），logits 不能保证 bit 精确。

本测试验证两点：
  1) 前向能跑通且 logits 形状一致
  2) 可训练参数量在同一量级（差距 < 2x）
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from common import count_parameters  # noqa: E402
from prefix_tuning_minimal import PrefixTuningGPT2  # noqa: E402
from prefix_tuning_peft import build_peft_model  # noqa: E402


def test_shape_and_param_order() -> None:
    torch.manual_seed(42)
    P = 10

    m_min = PrefixTuningGPT2(prefix_length=P, mid_dim=512).eval()
    m_peft = build_peft_model(prefix_length=P, mid_dim=512).eval()

    tok = m_min.tokenizer
    enc = tok("hello world", return_tensors="pt", padding=True)

    with torch.no_grad():
        out_min = m_min(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        out_peft = m_peft(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
        )

    # 1. 形状一致
    print(f"minimal logits: {out_min.logits.shape}")
    print(f"peft    logits: {out_peft.logits.shape}")
    assert out_min.logits.shape == out_peft.logits.shape, (
        f"Shape 不一致: minimal={out_min.logits.shape}, peft={out_peft.logits.shape}"
    )

    # 2. 可训练参数量
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
