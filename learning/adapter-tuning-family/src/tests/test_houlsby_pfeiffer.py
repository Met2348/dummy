"""Houlsby + Pfeiffer 一致性测试。

测试目标:
  1. 参数量符合预期 (Houlsby = 2x Pfeiffer)
  2. 初始 forward = base（up 层零初始化）
  3. mini training: loss 下降
  4. minimal vs adapters 库参数量一致
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.append(str(Path(__file__).parent.parent))
from houlsby_minimal import HoulsbyGPT2  # noqa: E402
from pfeiffer_minimal import PfeifferGPT2  # noqa: E402


def test_param_count_ratio():
    print("\n[Test 1] Houlsby 参数 = 2 × Pfeiffer 参数")
    torch.manual_seed(42)
    houlsby = HoulsbyGPT2(r=16)
    pfeiffer = PfeifferGPT2(r=16)
    n_h = sum(p.numel() for p in houlsby.parameters() if p.requires_grad)
    n_p = sum(p.numel() for p in pfeiffer.parameters() if p.requires_grad)
    print(f"  Houlsby trainable:  {n_h:,}")
    print(f"  Pfeiffer trainable: {n_p:,}")
    print(f"  ratio:              {n_h / n_p:.2f}")
    assert n_h == 2 * n_p, f"应为 2x 关系: {n_h} != 2*{n_p}"
    print("  [PASS]")


def test_initial_forward():
    print("\n[Test 2] 初始 forward = base GPT-2")
    from transformers import GPT2LMHeadModel
    torch.manual_seed(42)
    houlsby = HoulsbyGPT2(r=16).eval()
    pfeiffer = PfeifferGPT2(r=16).eval()
    base = GPT2LMHeadModel.from_pretrained("gpt2").eval()

    enc = houlsby.tokenizer("hello world this is a test", return_tensors="pt", padding=True)
    with torch.no_grad():
        out_h = houlsby(enc["input_ids"], enc["attention_mask"])
        out_p = pfeiffer(enc["input_ids"], enc["attention_mask"])
        out_b = base(enc["input_ids"], attention_mask=enc["attention_mask"])

    diff_h = (out_h.logits - out_b.logits).abs().max().item()
    diff_p = (out_p.logits - out_b.logits).abs().max().item()
    print(f"  Houlsby vs base:  max |Δlogits| = {diff_h:.4e}")
    print(f"  Pfeiffer vs base: max |Δlogits| = {diff_p:.4e}")
    assert diff_h < 1e-5
    assert diff_p < 1e-5
    print("  [PASS] 强一致（up 层零初始化）")


def test_mini_training():
    print("\n[Test 3] mini training: loss 下降")
    torch.manual_seed(42)
    model = HoulsbyGPT2(r=16)
    tok = model.tokenizer
    enc = tok(
        ["hello world this is a test", "i love this product",
         "absolutely terrible movie", "fantastic experience"],
        return_tensors="pt", padding=True,
    )
    enc["labels"] = enc["input_ids"].clone()

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    losses = []
    for _ in range(10):
        opt.zero_grad()
        out = model(**enc)
        out.loss.backward()
        opt.step()
        losses.append(out.loss.item())
    print(f"  loss: {losses[0]:.3f} → {losses[-1]:.3f}")
    assert losses[-1] < losses[0]
    print("  [PASS]")


def test_param_match_adapters_lib():
    print("\n[Test 4] minimal vs adapters 库参数量一致")
    pytest.importorskip("adapters")
    from houlsby_adapters import build_houlsby_model
    from pfeiffer_adapters import build_pfeiffer_model
    torch.manual_seed(42)
    minimal_h = HoulsbyGPT2(r=16)
    minimal_p = PfeifferGPT2(r=16)
    lib_h = build_houlsby_model(reduction_factor=48)
    lib_p = build_pfeiffer_model(reduction_factor=48)

    n_mh = sum(p.numel() for p in minimal_h.parameters() if p.requires_grad)
    n_mp = sum(p.numel() for p in minimal_p.parameters() if p.requires_grad)
    n_lh = sum(p.numel() for p in lib_h.parameters() if p.requires_grad)
    n_lp = sum(p.numel() for p in lib_p.parameters() if p.requires_grad)
    print(f"  Houlsby:  minimal={n_mh:,}  lib={n_lh:,}  diff={abs(n_mh-n_lh)}")
    print(f"  Pfeiffer: minimal={n_mp:,}  lib={n_lp:,}  diff={abs(n_mp-n_lp)}")
    assert n_mh == n_lh
    assert n_mp == n_lp
    print("  [PASS] 完美一致")


if __name__ == "__main__":
    test_param_count_ratio()
    test_initial_forward()
    test_mini_training()
    test_param_match_adapters_lib()
    print("\n" + "=" * 60)
    print("[PASS] Houlsby + Pfeiffer 全部测试通过")
    print("=" * 60)
