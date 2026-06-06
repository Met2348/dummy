"""AdapterFusion 测试。

测试目标:
  1. 参数量: 主要是 fusion 层 (12 × 3 × d²)
  2. 只有 fusion 可训练，adapter 应冻结
  3. mini training: loss 下降
  4. minimal vs adapters 库参数量近似一致
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.append(str(Path(__file__).parent.parent))
from adapterfusion_minimal import AdapterFusionGPT2  # noqa: E402


def test_only_fusion_trainable():
    print("\n[Test 1] 只有 fusion 层可训练")
    torch.manual_seed(42)
    model = AdapterFusionGPT2(n_adapters=3, r=16)

    # 数 fusion 参数量 vs total
    fusion_params = 0
    adapter_params = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "fusion" in name:
            fusion_params += p.numel()
        elif "adapter" in name:
            adapter_params += p.numel()

    print(f"  fusion 可训练:  {fusion_params:,}")
    print(f"  adapter 可训练: {adapter_params:,}")
    # adapter 应该全冻结
    assert adapter_params == 0, "adapter 应该冻结"
    # fusion 主要参数: 12 × 3 × 768² = 21,233,664
    expected_fusion = 12 * 3 * 768 * 768
    assert fusion_params == expected_fusion, f"{fusion_params} != {expected_fusion}"
    print(f"  expected: 12 * 3 * 768^2 = {expected_fusion:,}")
    print("  [PASS]")


def test_mini_training():
    print("\n[Test 2] mini training: loss 下降")
    torch.manual_seed(42)
    model = AdapterFusionGPT2(n_adapters=3, r=16)
    tok = model.tokenizer
    enc = tok(
        ["hello world this is a test", "i love this product",
         "absolutely terrible movie", "fantastic experience"],
        return_tensors="pt", padding=True,
    )
    enc["labels"] = enc["input_ids"].clone()

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
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


def test_param_count_match_lib():
    print("\n[Test 3] minimal vs adapters 库参数量（近似一致）")
    pytest.importorskip("adapters")
    from adapterfusion_adapters import build_fusion_model
    torch.manual_seed(42)
    minimal = AdapterFusionGPT2(n_adapters=3, r=16)
    lib = build_fusion_model(n_adapters=3, reduction_factor=48)

    n_m = sum(p.numel() for p in minimal.parameters() if p.requires_grad)
    n_l = sum(p.numel() for p in lib.parameters() if p.requires_grad)
    print(f"  minimal: {n_m:,}")
    print(f"  lib:     {n_l:,}")
    print(f"  diff:    {abs(n_m - n_l):,}  ({100 * abs(n_m-n_l)/n_l:.2f}%)")
    # adapters 库的 fusion 有额外 bias 等，允许 < 1% 差异
    assert abs(n_m - n_l) / n_l < 0.01
    print("  [PASS] 近似一致 (差异 < 1%)")


if __name__ == "__main__":
    test_only_fusion_trainable()
    test_mini_training()
    test_param_count_match_lib()
    print("\n" + "=" * 60)
    print("[PASS] AdapterFusion 全部测试通过")
    print("=" * 60)
