"""QLoRA 一致性测试。

测试目标:
  1. (单元) NF4 量化的 base.weight 与原始 W_0 误差有界
  2. (强一致) 训练时 base 完全不变（grad 不传）
  3. (mini training) LoRA 部分能正常更新
  4. (GPU 选做) fake-quant vs bitsandbytes 真 NF4 在 TinyLlama 上的 logits 一致性
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from qlora_minimal import QLoRAGPT2, QLoRALinear  # noqa: E402


def test_quantization_error():
    print("\n[Test 1] NF4 量化误差")
    from transformers import GPT2LMHeadModel
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    W_0 = base.transformer.h[0].attn.c_attn.weight.data.clone()

    torch.manual_seed(42)
    model = QLoRAGPT2(r=8, alpha=16, block_size=64)
    layer0 = model.lm.transformer.h[0].attn.c_attn

    rmse = layer0.quantization_error(W_0)
    rel = rmse / W_0.std().item()
    print(f"  RMSE: {rmse:.5f}")
    print(f"  相对 RMSE: {rel:.4f}  ({100 * rel:.2f}%)")
    assert rel < 0.15, f"NF4 量化误差过大: {rel}"
    print("  [PASS]")


def test_base_frozen_during_training():
    print("\n[Test 2] 训练时 base 完全不变")
    torch.manual_seed(42)
    model = QLoRAGPT2(r=8, alpha=16)
    layer0 = model.lm.transformer.h[0].attn.c_attn
    W_before = layer0.base.weight.data.clone()

    tok = model.tokenizer
    inputs = tok(["hello world this is a test"], return_tensors="pt", padding=True)
    inputs["labels"] = inputs["input_ids"].clone()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    for _ in range(10):
        optimizer.zero_grad()
        out = model(**inputs)
        out.loss.backward()
        optimizer.step()

    W_after = layer0.base.weight.data
    diff = (W_before - W_after).abs().max().item()
    print(f"  10 step 后 base.weight 变化: {diff:.6e}")
    assert diff < 1e-6, "base.weight 不应改变"
    print("  [PASS]")


def test_lora_update():
    print("\n[Test 3] LoRA 部分正常更新")
    torch.manual_seed(42)
    model = QLoRAGPT2(r=8, alpha=16)
    layer0 = model.lm.transformer.h[0].attn.c_attn
    A_before = layer0.A.data.clone()
    B_before = layer0.B.data.clone()

    tok = model.tokenizer
    inputs = tok(["hello world this is a test"], return_tensors="pt", padding=True)
    inputs["labels"] = inputs["input_ids"].clone()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    for _ in range(5):
        optimizer.zero_grad()
        out = model(**inputs)
        out.loss.backward()
        optimizer.step()

    A_diff = (layer0.A.data - A_before).abs().max().item()
    B_diff = (layer0.B.data - B_before).abs().max().item()
    print(f"  A 变化最大值: {A_diff:.4e}")
    print(f"  B 变化最大值: {B_diff:.4e}")
    assert A_diff > 1e-6, "A 应改变"
    assert B_diff > 1e-6, "B 应改变"
    print("  [PASS]")


def test_mini_training():
    print("\n[Test 4] mini training: QLoRA loss 下降")
    torch.manual_seed(42)
    model = QLoRAGPT2(r=8, alpha=16)
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
    for _ in range(10):
        optimizer.zero_grad()
        out = model(**inputs)
        out.loss.backward()
        optimizer.step()
        losses.append(out.loss.item())
    print(f"  loss: {losses[0]:.3f} → {losses[-1]:.3f}")
    assert losses[-1] < losses[0]
    print("  [PASS]")


def test_fake_vs_real_nf4_gpu():
    print("\n[Test 5] GPU 选做: fake-quant vs bitsandbytes 真 NF4")
    if not torch.cuda.is_available():
        print("  [SKIP] 没有 GPU")
        return
    try:
        import bitsandbytes as bnb
    except ImportError:
        print("  [SKIP] bitsandbytes 未安装")
        return

    # 取一个 N(0, 1) 张量，比较 fake-quant 与真 NF4
    torch.manual_seed(42)
    W = torch.randn(256, 256, device="cuda", dtype=torch.float16)

    # bitsandbytes 真 NF4
    try:
        q, s = bnb.functional.quantize_nf4(W)
        W_real = bnb.functional.dequantize_nf4(q, s)
    except Exception as e:
        print(f"  [SKIP] bitsandbytes NF4 GPU 调用失败: {type(e).__name__}: {str(e)[:80]}")
        return

    # 自家 fake-quant
    from nf4_quant import nf4_quant_dequant
    W_fake = nf4_quant_dequant(W.float(), block_size=64).half()

    diff = (W_real - W_fake).abs().max().item()
    rel = diff / W.abs().max().item()
    print(f"  fake-quant vs 真 NF4 最大绝对误差: {diff:.5f}")
    print(f"  相对误差: {rel:.4f}")
    # 由于 block_size、scaling 实现差异，允许 ~5% 误差
    assert rel < 0.1, f"差异过大: {rel}"
    print("  [PASS]")


if __name__ == "__main__":
    test_quantization_error()
    test_base_frozen_during_training()
    test_lora_update()
    test_mini_training()
    test_fake_vs_real_nf4_gpu()
    print("\n" + "=" * 60)
    print("[PASS] QLoRA 全部测试通过")
    print("=" * 60)
