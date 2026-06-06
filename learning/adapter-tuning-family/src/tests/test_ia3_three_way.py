"""(IA)³ 三轨一致性测试（minimal vs adapters lib vs peft）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.append(str(Path(__file__).parent.parent))
from ia3_minimal import IA3GPT2  # noqa: E402


def test_minimal_param_count():
    print("\n[Test 1] minimal 参数量 = 55,296")
    torch.manual_seed(42)
    model = IA3GPT2()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = 12 * (768 + 768 + 3072)
    print(f"  trainable: {n:,}")
    print(f"  expected:  {expected:,}")
    assert n == expected
    print("  [PASS]")


def test_three_way_param_match():
    print("\n[Test 2] 三轨参数量一致")
    pytest.importorskip("adapters")
    from ia3_adapters import build_ia3_model_adapters
    from ia3_peft import build_ia3_model_peft
    torch.manual_seed(42)
    m = IA3GPT2()
    torch.manual_seed(42)
    a = build_ia3_model_adapters()
    torch.manual_seed(42)
    p_model = build_ia3_model_peft()

    n_m = sum(x.numel() for x in m.parameters() if x.requires_grad)
    n_a = sum(x.numel() for x in a.parameters() if x.requires_grad)
    n_p = sum(x.numel() for x in p_model.parameters() if x.requires_grad)
    print(f"  minimal:     {n_m:,}")
    print(f"  adapters库:  {n_a:,}")
    print(f"  peft:        {n_p:,}")
    assert n_m == n_a == n_p
    print("  [PASS] 三方完全一致 (但实现策略不同)")


def test_initial_forward_equals_base():
    print("\n[Test 3] 三方初始 forward = base GPT-2")
    pytest.importorskip("adapters")
    from transformers import GPT2LMHeadModel
    base = GPT2LMHeadModel.from_pretrained("gpt2").eval()

    torch.manual_seed(42)
    m = IA3GPT2().eval()
    torch.manual_seed(42)
    from ia3_adapters import build_ia3_model_adapters
    a = build_ia3_model_adapters().eval()
    torch.manual_seed(42)
    from ia3_peft import build_ia3_model_peft
    p_model = build_ia3_model_peft().eval()

    enc = m.tokenizer("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out_m = m(enc["input_ids"], enc["attention_mask"]).logits
        # adapters lib 需要先 activate
        a.set_active_adapters("demo")
        out_a = a(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        out_p = p_model(enc["input_ids"], enc["attention_mask"]).logits
        out_b = base(enc["input_ids"], attention_mask=enc["attention_mask"]).logits

    for name, out in [("minimal", out_m), ("adapters", out_a), ("peft", out_p)]:
        diff = (out - out_b).abs().max().item()
        print(f"  {name:<10} vs base max|Δ|: {diff:.4e}")
        assert diff < 1e-4, f"{name} should match base at init"
    print("  [PASS]")


def test_mini_training():
    print("\n[Test 4] (IA)^3 mini training: loss 下降")
    torch.manual_seed(42)
    model = IA3GPT2()
    tok = model.tokenizer
    enc = tok(
        ["hello world this is a test", "i love this product",
         "absolutely terrible movie", "fantastic experience"],
        return_tensors="pt", padding=True,
    )
    enc["labels"] = enc["input_ids"].clone()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2
    )
    losses = []
    for _ in range(10):
        opt.zero_grad()
        out = model(**enc)
        out.loss.backward()
        opt.step()
        losses.append(out.loss.item())
    print(f"  loss: {losses[0]:.3f} -> {losses[-1]:.3f}")
    assert losses[-1] < losses[0]
    print("  [PASS]")


if __name__ == "__main__":
    test_minimal_param_count()
    test_three_way_param_match()
    test_initial_forward_equals_base()
    test_mini_training()
    print("\n" + "=" * 60)
    print("[PASS] (IA)^3 三轨测试通过")
    print("=" * 60)
