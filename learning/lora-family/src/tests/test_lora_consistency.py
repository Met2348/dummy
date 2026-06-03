"""LoRA minimal 与 peft 强一致性测试。

minimal: A (r, d_in), B (d_out, r)，forward: x @ A.T @ B.T
peft:    lora_A.weight (r, d_in), lora_B.weight (d_out, r)，forward 等价

LoRA 无 reparam、无 LSTM、无随机 dropout（设 dropout=0），
理论上 minimal 与 peft 的 logits 应**完全一致**（bit 精确）。

策略：
  1. 探测 peft 的 lora_A、lora_B 参数位置
  2. 按 layer index 把 minimal 的 A、B copy 过去
  3. 双方前向，比较 logits
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from lora_minimal import LoRAGPT2, LoRALinear  # noqa: E402
from lora_peft import build_peft_model  # noqa: E402


def test_logits_match() -> None:
    torch.manual_seed(42)

    m_min = LoRAGPT2(r=8, alpha=16, dropout=0.0).eval()
    m_peft = build_peft_model(r=8, alpha=16, dropout=0.0).eval()

    # 1. 探测 peft 的 LoRA 参数（按层 index 索引）
    print("peft 可训练参数（按层组织）：")
    peft_loras = {}
    for name, p in m_peft.named_parameters():
        if not p.requires_grad:
            continue
        parts = name.split(".")
        try:
            h_idx = parts.index("h")
            layer_idx = int(parts[h_idx + 1])
        except (ValueError, IndexError):
            continue
        ab = "A" if "lora_A" in name else ("B" if "lora_B" in name else None)
        if ab is None:
            continue
        peft_loras.setdefault(layer_idx, {})[ab] = p

    print(f"  探测到 {len(peft_loras)} 层 LoRA 参数")
    sample = peft_loras[0]
    print(f"  layer 0 A shape: {tuple(sample['A'].shape)}")
    print(f"  layer 0 B shape: {tuple(sample['B'].shape)}")
    assert len(peft_loras) == 12, f"应该有 12 层，实际 {len(peft_loras)}"

    # 2. 探测 minimal 的 LoRA 模块（按层 index 索引）
    minimal_loras = {}
    for name, mod in m_min.named_modules():
        if isinstance(mod, LoRALinear):
            parts = name.split(".")
            try:
                h_idx = parts.index("h")
                layer_idx = int(parts[h_idx + 1])
            except (ValueError, IndexError):
                continue
            minimal_loras[layer_idx] = mod
    assert len(minimal_loras) == 12

    # 3. 把 minimal 的 A、B copy 到 peft
    print("\n复制 minimal A、B 到 peft 中...")
    with torch.no_grad():
        for idx in range(12):
            assert minimal_loras[idx].A.shape == peft_loras[idx]["A"].shape, (
                f"layer {idx} A shape 不匹配"
            )
            assert minimal_loras[idx].B.shape == peft_loras[idx]["B"].shape, (
                f"layer {idx} B shape 不匹配"
            )
            peft_loras[idx]["A"].copy_(minimal_loras[idx].A)
            peft_loras[idx]["B"].copy_(minimal_loras[idx].B)
    print("  [OK] 12 层 LoRA 参数已对齐")

    # 4. 双方前向
    tok = m_min.tokenizer
    enc = tok("hello world", return_tensors="pt", padding=True)

    with torch.no_grad():
        out_min = m_min(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        out_peft = m_peft(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

    print(f"\nminimal logits: {tuple(out_min.logits.shape)}")
    print(f"peft    logits: {tuple(out_peft.logits.shape)}")
    assert out_min.logits.shape == out_peft.logits.shape

    diff = (out_min.logits - out_peft.logits).abs().max().item()
    rel_diff = diff / out_min.logits.abs().max().item()
    print(f"\nlogits 最大绝对误差: {diff:.4e}")
    print(f"logits 最大相对误差: {rel_diff:.4e}")

    # LoRA 无 reparam → 强一致（容差仅为浮点 round-off）
    assert diff < 1e-3, f"Logits 差异过大: {diff}"
    print("[PASS] minimal 与 peft 输出强一致")


if __name__ == "__main__":
    test_logits_match()
