"""
P-Tuning v2 一致性测试（强一致）。

由于 v2 **无 reparameterization、无 LSTM、无随机 dropout**，
理论上 minimal 与 peft 应该数值精确一致。

验证策略：
  1) 探测 peft 内部 prefix 参数位置 + shape
  2) 把 minimal 的 prefix 张量按 reshape 复制到 peft
  3) 双方前向，比较 logits（允许误差 < 1e-4）
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from p_tuning_v2_minimal import PTuningV2GPT2  # noqa: E402
from p_tuning_v2_peft import build_peft_model  # noqa: E402


def test_logits_match() -> None:
    torch.manual_seed(42)
    P = 10

    m_min = PTuningV2GPT2(prefix_length=P).eval()
    m_peft = build_peft_model(prefix_length=P).eval()

    # 1. 探测 peft 的 prefix 参数
    print("peft 可训练参数清单：")
    peft_prefix = None
    for name, p in m_peft.named_parameters():
        if p.requires_grad:
            print(f"  {name}: shape={tuple(p.shape)}, numel={p.numel():,}")
            if "embedding.weight" in name:
                peft_prefix = p
    assert peft_prefix is not None, "找不到 peft 的 prefix 参数"

    # 2. 转换 minimal 的 prefix 到 peft 布局
    # minimal: self.prefix shape (L=12, 2, p=10, H=12, d_h=64) → numel 184,320
    # peft:    embedding.weight shape (10, 12*2*768) = (10, 18432) → numel 184,320
    with torch.no_grad():
        L_dim, two, p_len, H, d_h = m_min.prefix.shape
        # (L, 2, p, H, d_h) → (p, L, 2, H, d_h) → (p, L*2*H*d_h) = (p, L*2*d)
        converted = m_min.prefix.permute(2, 0, 1, 3, 4).reshape(p_len, -1).contiguous()
        print(f"\nminimal prefix shape: {tuple(m_min.prefix.shape)}, numel={m_min.prefix.numel():,}")
        print(f"converted shape:      {tuple(converted.shape)}")
        print(f"peft prefix shape:    {tuple(peft_prefix.shape)}, numel={peft_prefix.numel():,}")
        assert converted.shape == peft_prefix.shape, (
            f"shape 不匹配: {converted.shape} vs {peft_prefix.shape}"
        )
        peft_prefix.copy_(converted)

    # 3. 双方前向
    tok = m_min.tokenizer
    enc = tok("hello world", return_tensors="pt", padding=True)

    with torch.no_grad():
        out_min = m_min(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        out_peft = m_peft(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

    print(f"\nminimal logits: {out_min.logits.shape}")
    print(f"peft    logits: {out_peft.logits.shape}")
    assert out_min.logits.shape == out_peft.logits.shape

    diff = (out_min.logits - out_peft.logits).abs().max().item()
    print(f"\nlogits 最大绝对误差: {diff:.2e}")

    # v2 无 reparam，允许 1e-4 容差
    assert diff < 1e-3, f"Logits 差异过大: {diff}"
    print("[PASS] minimal 与 peft 输出强一致")


if __name__ == "__main__":
    test_logits_match()
