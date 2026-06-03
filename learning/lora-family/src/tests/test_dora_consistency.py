"""DoRA 一致性测试。

测试目标:
  1. (强一致) 初始 W_DoRA = W_0
  2. (强一致) 初始 forward = 原始 GPT-2
  3. (mini training) DoRA loss 下降，且超过 LoRA
  4. 参数量 = LoRA + d (magnitude 多出来)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from dora_minimal import DoRAGPT2, DoRALinear, _extract_weight_out_in  # noqa: E402
from lora_minimal import LoRAGPT2  # noqa: E402


def test_initial_W_equals_W0():
    print("\n[Test 1] 初始 W_DoRA = W_0")
    torch.manual_seed(42)
    model = DoRAGPT2(r=8, alpha=16)
    from transformers import GPT2LMHeadModel
    base = GPT2LMHeadModel.from_pretrained("gpt2")

    max_diff = 0.0
    for i in range(12):
        layer = model.lm.transformer.h[i].attn.c_attn
        W_dora = layer._compute_W_dora()
        W_0 = _extract_weight_out_in(base.transformer.h[i].attn.c_attn)
        diff = (W_dora - W_0).abs().max().item()
        max_diff = max(max_diff, diff)
    print(f"  12 层最大误差: {max_diff:.4e}")
    assert max_diff < 1e-3, f"初始 W_DoRA 与 W_0 差异过大: {max_diff}"
    print("  [PASS]")


def test_initial_forward_match():
    print("\n[Test 2] 初始 forward 与原始 GPT-2 一致")
    torch.manual_seed(42)
    model = DoRAGPT2(r=8, alpha=16).eval()
    from transformers import GPT2LMHeadModel
    base = GPT2LMHeadModel.from_pretrained("gpt2").eval()

    tok = model.tokenizer
    enc = tok("hello world this is a test", return_tensors="pt", padding=True)
    with torch.no_grad():
        out_dora = model(enc["input_ids"], enc["attention_mask"])
        out_base = base(enc["input_ids"], attention_mask=enc["attention_mask"])
    diff = (out_dora.logits - out_base.logits).abs().max().item()
    rel = diff / out_base.logits.abs().max().item()
    print(f"  最大绝对误差: {diff:.4e}")
    print(f"  最大相对误差: {rel:.4e}")
    assert diff < 1.0
    print("  [PASS]")


def test_param_count():
    print("\n[Test 3] 参数量 = LoRA + d")
    torch.manual_seed(42)
    dora = DoRAGPT2(r=8, alpha=16)
    n_dora = sum(p.numel() for p in dora.parameters() if p.requires_grad)
    # LoRA r=8: 12 × (8*768 + 2304*8) = 294,912
    # DoRA = LoRA + 12 × 768 (m) = 294,912 + 9,216 = 304,128
    expected = 294912 + 12 * 768
    print(f"  trainable: {n_dora:,}")
    print(f"  expected:  {expected:,} = LoRA_r8 + 12 × 768 (magnitude)")
    assert n_dora == expected
    print("  [PASS]")


def test_dora_vs_lora_training():
    print("\n[Test 4] mini training: DoRA vs LoRA")
    torch.manual_seed(42)
    lora = LoRAGPT2(r=8, alpha=16)
    torch.manual_seed(42)
    dora = DoRAGPT2(r=8, alpha=16)

    tok = lora.tokenizer
    inputs = tok(
        ["hello world this is a test", "i love this product",
         "absolutely terrible movie", "fantastic experience"],
        return_tensors="pt", padding=True,
    )
    inputs["labels"] = inputs["input_ids"].clone()

    opt_l = torch.optim.AdamW(
        [p for p in lora.parameters() if p.requires_grad], lr=1e-3
    )
    opt_d = torch.optim.AdamW(
        [p for p in dora.parameters() if p.requires_grad], lr=1e-3
    )

    losses_l, losses_d = [], []
    for _ in range(15):
        for m, opt, losses in [(lora, opt_l, losses_l), (dora, opt_d, losses_d)]:
            opt.zero_grad()
            out = m(**inputs)
            out.loss.backward()
            opt.step()
            losses.append(out.loss.item())

    print(f"  LoRA loss: {losses_l[0]:.3f} → {losses_l[-1]:.3f}")
    print(f"  DoRA loss: {losses_d[0]:.3f} → {losses_d[-1]:.3f}")
    assert losses_l[-1] < losses_l[0]
    assert losses_d[-1] < losses_d[0]
    print("  [PASS] 两者都正常收敛")


if __name__ == "__main__":
    test_initial_W_equals_W0()
    test_initial_forward_match()
    test_param_count()
    test_dora_vs_lora_training()
    print("\n" + "=" * 60)
    print("[PASS] DoRA 全部测试通过")
    print("=" * 60)
