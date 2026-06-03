"""MAM Adapter 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from mam_minimal import MAMGPT2  # noqa: E402


def test_param_count():
    print("\n[Test 1] MAM 参数量 = Prefix + Parallel")
    torch.manual_seed(42)
    model = MAMGPT2(prefix_len=30, r=16)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 每层: 2*30*768 (prefix) + 2*768*16 + 16 + 768 (parallel)
    per_layer = 2 * 30 * 768 + 2 * 768 * 16 + 16 + 768
    expected = 12 * per_layer
    print(f"  trainable: {n:,}")
    print(f"  expected:  {expected:,} = 12 * ({2*30*768} + {2*768*16 + 16 + 768})")
    assert n == expected
    print("  [PASS]")


def test_mini_training():
    print("\n[Test 2] MAM mini training")
    torch.manual_seed(42)
    model = MAMGPT2(prefix_len=30, r=16)
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
    print(f"  loss: {losses[0]:.3f} -> {losses[-1]:.3f}")
    assert losses[-1] < losses[0]
    print("  [PASS]")


def test_components_independent():
    print("\n[Test 3] Prefix 和 Parallel 是独立组件")
    torch.manual_seed(42)
    model = MAMGPT2(prefix_len=30, r=16)
    # 检查每个 block 都有 PrefixAttention 和 ParallelAdapter
    from mam_minimal import PrefixAttention
    from parallel_minimal import ParallelAdapter
    for i, block in enumerate(model.lm.transformer.h):
        assert isinstance(block.attn.c_attn, PrefixAttention), f"block {i}: missing PrefixAttention"
        assert isinstance(block.mlp, ParallelAdapter), f"block {i}: missing ParallelAdapter"
    print("  12 个 block 各有 Prefix + Parallel")
    print("  [PASS]")


if __name__ == "__main__":
    test_param_count()
    test_mini_training()
    test_components_independent()
    print("\n" + "=" * 60)
    print("[PASS] MAM Adapter 全部测试通过")
    print("=" * 60)
