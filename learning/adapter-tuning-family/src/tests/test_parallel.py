"""Parallel Adapter 测试。

测试目标:
  1. 参数量 = Pfeiffer (因为只是改了结构非参数大小)
  2. 初始 forward = base
  3. mini training
  4. minimal vs adapters 库
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from parallel_minimal import ParallelAdapterGPT2, ParallelAdapter  # noqa: E402
from pfeiffer_minimal import PfeifferGPT2  # noqa: E402


def test_param_count():
    print("\n[Test 1] Parallel 参数量 = Pfeiffer")
    torch.manual_seed(42)
    par = ParallelAdapterGPT2(r=16)
    pfe = PfeifferGPT2(r=16)
    n_par = sum(p.numel() for p in par.parameters() if p.requires_grad)
    n_pfe = sum(p.numel() for p in pfe.parameters() if p.requires_grad)
    print(f"  Parallel:  {n_par:,}")
    print(f"  Pfeiffer:  {n_pfe:,}")
    assert n_par == n_pfe
    print("  [PASS]")


def test_initial_forward():
    print("\n[Test 2] Parallel 初始 forward = base")
    from transformers import GPT2LMHeadModel
    torch.manual_seed(42)
    model = ParallelAdapterGPT2(r=16).eval()
    base = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    enc = model.tokenizer("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out_p = model(enc["input_ids"], enc["attention_mask"])
        out_b = base(enc["input_ids"], attention_mask=enc["attention_mask"])
    diff = (out_p.logits - out_b.logits).abs().max().item()
    print(f"  max |Δlogits|: {diff:.4e}")
    assert diff < 1e-5
    print("  [PASS]")


def test_serial_vs_parallel_diff():
    """串联 vs 并联：训练后会得到不同的权重（结构本质不同）。"""
    print("\n[Test 3] 串联 vs 并联：训练后权重不同")
    torch.manual_seed(42)
    pfe = PfeifferGPT2(r=16)
    torch.manual_seed(42)
    par = ParallelAdapterGPT2(r=16)

    tok = pfe.tokenizer
    enc = tok(
        ["hello world", "i love this"],
        return_tensors="pt", padding=True,
    )
    enc["labels"] = enc["input_ids"].clone()

    for model in [pfe, par]:
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        for _ in range(5):
            opt.zero_grad()
            out = model(**enc)
            out.loss.backward()
            opt.step()

    # 取第 0 层的 down 矩阵对比
    pfe_down = pfe.lm.transformer.h[0].mlp.adapter.down.weight.data
    par_down = par.lm.transformer.h[0].mlp.down.weight.data
    diff = (pfe_down - par_down).abs().mean().item()
    print(f"  Pfeiffer vs Parallel 第 0 层 down 权重差异: {diff:.4e}")
    assert diff > 1e-5, "应该不同（结构不同导致梯度不同）"
    print("  [PASS] 结构差异导致训练出不同的权重")


def test_param_match_lib():
    print("\n[Test 4] minimal vs adapters 库")
    from parallel_adapters import build_parallel_model
    torch.manual_seed(42)
    minimal = ParallelAdapterGPT2(r=16)
    lib = build_parallel_model(reduction_factor=48)
    n_m = sum(p.numel() for p in minimal.parameters() if p.requires_grad)
    n_l = sum(p.numel() for p in lib.parameters() if p.requires_grad)
    print(f"  minimal: {n_m:,}")
    print(f"  lib:     {n_l:,}")
    assert n_m == n_l
    print("  [PASS] 完美一致")


if __name__ == "__main__":
    test_param_count()
    test_initial_forward()
    test_serial_vs_parallel_diff()
    test_param_match_lib()
    print("\n" + "=" * 60)
    print("[PASS] Parallel Adapter 全部测试通过")
    print("=" * 60)
