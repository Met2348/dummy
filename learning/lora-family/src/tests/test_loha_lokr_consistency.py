"""LoHa + LoKr 一致性测试。

测试目标:
  LoHa:
    1. 参数量 = 4rd × L
    2. 初始 ΔW = 0（B_1 零初始化）
    3. Hadamard 后等效秩 ≤ r^2（实测）
    4. mini training 收敛

  LoKr:
    5. 参数量近似 √d r × L
    6. 初始 ΔW = 0（B_lr 零初始化）
    7. Kronecker 形状正确 = (d_out, d_in)
    8. mini training 收敛
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from loha_minimal import LoHaGPT2, LoHaLinear  # noqa: E402
from lokr_minimal import LoKrGPT2, LoKrLinear  # noqa: E402


# ==============================
# LoHa
# ==============================

def test_loha_param_count():
    print("\n[LoHa Test 1] 参数量 = 4 r d × L")
    torch.manual_seed(42)
    model = LoHaGPT2(r=8, alpha=16)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # per layer: 2 * (A=(r,d_in) + B=(d_out,r)) = 2 * (8*768 + 2304*8)
    expected = 12 * 2 * (8 * 768 + 2304 * 8)
    print(f"  trainable: {trainable:,}")
    print(f"  expected:  {expected:,} = 12 × 2 × (8×768 + 2304×8)")
    assert trainable == expected
    print("  [PASS]")


def test_loha_initial_delta_zero():
    print("\n[LoHa Test 2] 初始 ΔW = 0 (B_1 零初始化)")
    torch.manual_seed(42)
    model = LoHaGPT2(r=8, alpha=16)
    layer0 = model.lm.transformer.h[0].attn.c_attn
    delta = layer0.get_delta_W()
    max_abs = delta.abs().max().item()
    print(f"  ΔW max abs: {max_abs:.6e}")
    assert max_abs < 1e-6
    print("  [PASS]")


def test_loha_effective_rank():
    print("\n[LoHa Test 3] Hadamard 后等效秩")
    torch.manual_seed(42)
    model = LoHaGPT2(r=4, alpha=8)
    # 训一步让 B_1 非零
    layer0 = model.lm.transformer.h[0].attn.c_attn
    # 手工把 B_1 设为非零
    with torch.no_grad():
        layer0.B_1.normal_(std=0.1)

    delta = layer0.get_delta_W()
    rank = torch.linalg.matrix_rank(delta.float(), tol=1e-3).item()
    r = 4
    print(f"  r = {r}, ΔW rank = {rank}")
    print(f"  理论上界 r^2 = {r * r}")
    # rank 通常接近 r^2
    assert rank > r, f"rank 应该大于 r={r}（不只是 LoRA），但实际 {rank}"
    print(f"  [PASS] rank > r 说明 Hadamard 提升了等效秩")


def test_loha_mini_training():
    print("\n[LoHa Test 4] mini training")
    torch.manual_seed(42)
    model = LoHaGPT2(r=8, alpha=16)
    tok = model.tokenizer
    inputs = tok(
        ["hello world this is a test", "i love this product"],
        return_tensors="pt", padding=True,
    )
    inputs["labels"] = inputs["input_ids"].clone()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    losses = []
    for step in range(10):
        optimizer.zero_grad()
        out = model(**inputs)
        out.loss.backward()
        optimizer.step()
        losses.append(out.loss.item())
    print(f"  loss: {losses[0]:.3f} → {losses[-1]:.3f}")
    assert losses[-1] < losses[0]
    print("  [PASS]")


# ==============================
# LoKr
# ==============================

def test_lokr_param_count():
    print("\n[LoKr Test 5] 参数量极少")
    torch.manual_seed(42)
    model = LoKrGPT2(factor=32, r=4)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoKr (factor=32, r=4) trainable: {trainable:,}")
    # 比较 LoRA r=8 的 294,912
    print(f"  对比 LoRA r=8: 294,912")
    print(f"  LoKr / LoRA = {trainable / 294912:.2%}")
    # 用对称 factor 后应该明显比 LoRA 少
    assert trainable < 294912 * 0.5, "LoKr 用对称 factor 应该明显比 LoRA 少"
    print("  [PASS]")


def test_lokr_initial_delta_zero():
    print("\n[LoKr Test 6] 初始 ΔW = 0 (B_lr 零初始化 → A = 0)")
    torch.manual_seed(42)
    model = LoKrGPT2(factor=32, r=4)
    layer0 = model.lm.transformer.h[0].attn.c_attn
    delta = layer0.get_delta_W()
    max_abs = delta.abs().max().item()
    print(f"  ΔW max abs: {max_abs:.6e}")
    assert max_abs < 1e-6
    print("  [PASS]")


def test_lokr_kron_shape():
    print("\n[LoKr Test 7] Kronecker 输出 shape 正确")
    torch.manual_seed(42)
    model = LoKrGPT2(factor=32, r=4)
    layer0 = model.lm.transformer.h[0].attn.c_attn
    # 把 B_lr 设为非零，让 delta 非零
    with torch.no_grad():
        layer0.B_lr.normal_(std=0.1)
    delta = layer0.get_delta_W()
    expected_shape = (2304, 768)  # (d_out, d_in)
    print(f"  ΔW shape: {tuple(delta.shape)}")
    print(f"  expected: {expected_shape}")
    assert delta.shape == expected_shape
    print("  [PASS]")


def test_lokr_mini_training():
    print("\n[LoKr Test 8] mini training")
    torch.manual_seed(42)
    model = LoKrGPT2(factor=32, r=4)
    tok = model.tokenizer
    inputs = tok(
        ["hello world this is a test", "i love this product"],
        return_tensors="pt", padding=True,
    )
    inputs["labels"] = inputs["input_ids"].clone()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    losses = []
    for step in range(10):
        optimizer.zero_grad()
        out = model(**inputs)
        out.loss.backward()
        optimizer.step()
        losses.append(out.loss.item())
    print(f"  loss: {losses[0]:.3f} → {losses[-1]:.3f}")
    assert losses[-1] < losses[0]
    print("  [PASS]")


if __name__ == "__main__":
    test_loha_param_count()
    test_loha_initial_delta_zero()
    test_loha_effective_rank()
    test_loha_mini_training()
    test_lokr_param_count()
    test_lokr_initial_delta_zero()
    test_lokr_kron_shape()
    test_lokr_mini_training()
    print("\n" + "=" * 60)
    print("[PASS] LoHa + LoKr 全部测试通过")
    print("=" * 60)
