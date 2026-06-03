"""VeRA 一致性测试。

测试目标:
  1. 参数量 = L × (r + d)
  2. 共享 A、B 跨层一致（指针相同 → 内存共享）
  3. 初始 Λ_b = 1, Λ_d = d_initial
  4. mini training 收敛
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from vera_minimal import VeRAGPT2, VeRALinear, VeRASharedBuffers  # noqa: E402


def test_parameter_count():
    print("\n[Test 1] 参数量 = L × (r + d)")
    torch.manual_seed(42)
    model = VeRAGPT2(r=256)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = 12 * (256 + 2304)  # L=12, r=256, d_out=2304 (c_attn)
    print(f"  trainable: {trainable:,}")
    print(f"  expected:  {expected:,} = 12 × (256 + 2304)")
    assert trainable == expected
    print("  [PASS]")


def test_shared_A_B():
    print("\n[Test 2] 共享 A、B 跨层一致")
    torch.manual_seed(42)
    model = VeRAGPT2(r=256)
    # 取 layer 0 和 layer 11 的 forward 中用到的 A
    layer0 = model.lm.transformer.h[0].attn.c_attn
    layer11 = model.lm.transformer.h[11].attn.c_attn

    A0 = VeRASharedBuffers.get_A(layer0.r, layer0.d_in)
    A11 = VeRASharedBuffers.get_A(layer11.r, layer11.d_in)
    # 应该是同一个 tensor 的 slice → data_ptr 相同
    assert A0.data_ptr() == A11.data_ptr(), "A 应该在所有层共享同一存储"
    print(f"  layer 0 与 layer 11 的 A 共享存储: data_ptr 相同  [OK]")
    print("  [PASS]")


def test_init_values():
    print("\n[Test 3] 初始 Λ_b = 1, Λ_d = 0.1")
    torch.manual_seed(42)
    model = VeRAGPT2(r=256, d_initial=0.1)
    layer0 = model.lm.transformer.h[0].attn.c_attn
    assert torch.allclose(layer0.Lambda_b, torch.ones(256)), "Λ_b 应该全 1"
    assert torch.allclose(layer0.Lambda_d, torch.full((2304,), 0.1)), "Λ_d 应该全 0.1"
    print(f"  Λ_b: 全 1  [OK]")
    print(f"  Λ_d: 全 0.1  [OK]")
    print("  [PASS]")


def test_mini_training():
    print("\n[Test 4] mini training 收敛")
    torch.manual_seed(42)
    model = VeRAGPT2(r=256, d_initial=0.1)

    tok = model.tokenizer
    inputs = tok(
        ["hello world this is a test",
         "i love this amazing wonderful product",
         "absolutely terrible movie do not buy"],
        return_tensors="pt", padding=True,
    )
    inputs["labels"] = inputs["input_ids"].clone()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2
    )
    losses = []
    for step in range(15):
        optimizer.zero_grad()
        out = model(**inputs)
        out.loss.backward()
        optimizer.step()
        losses.append(out.loss.item())

    print(f"  loss 轨迹: {losses[0]:.3f} → {losses[7]:.3f} → {losses[-1]:.3f}")
    assert losses[-1] < losses[0], "loss 没下降"
    print("  [PASS]")


if __name__ == "__main__":
    test_parameter_count()
    test_shared_A_B()
    test_init_values()
    test_mini_training()
    print("\n" + "=" * 60)
    print("[PASS] VeRA 全部测试通过")
    print("=" * 60)
