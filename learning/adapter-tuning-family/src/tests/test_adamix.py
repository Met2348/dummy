"""AdaMix 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from adamix_minimal import AdaMixGPT2  # noqa: E402


def test_param_count():
    print("\n[Test 1] AdaMix 参数 = N × 12 × per_layer")
    torch.manual_seed(42)
    m = AdaMixGPT2(r=16, n_experts=4)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    expected = 4 * 12 * (2 * 768 * 16 + 16 + 768)
    print(f"  trainable: {n:,}  expected: {expected:,}")
    assert n == expected
    print("  [PASS]")


def test_training_stochastic():
    print("\n[Test 2] 训练时随机路由 → 多次 forward loss 不同")
    torch.manual_seed(42)
    m = AdaMixGPT2(r=16, n_experts=4)
    m.train()
    tok = m.tokenizer
    enc = tok("hello world", return_tensors="pt", padding=True)
    enc["labels"] = enc["input_ids"].clone()

    torch.manual_seed(123)
    losses = []
    with torch.no_grad():
        for _ in range(5):
            losses.append(m(**enc).loss.item())
    print(f"  5 次 forward losses: {[f'{l:.3f}' for l in losses]}")
    # 至少 2 个不同值（不可能 5 次都选同一个）
    assert len(set([round(l, 3) for l in losses])) > 1
    print("  [PASS] 随机路由生效")


def test_inference_deterministic():
    print("\n[Test 3] 推理时 averaging → deterministic")
    torch.manual_seed(42)
    m = AdaMixGPT2(r=16, n_experts=4).eval()
    tok = m.tokenizer
    enc = tok("hello world", return_tensors="pt", padding=True)
    enc["labels"] = enc["input_ids"].clone()

    with torch.no_grad():
        l1 = m(**enc).loss.item()
        l2 = m(**enc).loss.item()
        l3 = m(**enc).loss.item()
    print(f"  3 次 forward losses: {l1:.4f} == {l2:.4f} == {l3:.4f}")
    assert l1 == l2 == l3
    print("  [PASS] averaging deterministic")


def test_merge_experts():
    print("\n[Test 4] merge_experts: N experts → 1 expert")
    torch.manual_seed(42)
    m = AdaMixGPT2(r=16, n_experts=4)
    n_before = sum(p.numel() for p in m.parameters() if p.requires_grad)
    m.merge_experts()
    n_after = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  合并前: {n_before:,}")
    print(f"  合并后: {n_after:,}")
    assert n_after == n_before // 4
    print("  [PASS] 参数 1/4 (单 expert)")


def test_mini_training():
    print("\n[Test 5] AdaMix mini training")
    torch.manual_seed(42)
    m = AdaMixGPT2(r=16, n_experts=4)
    tok = m.tokenizer
    enc = tok(
        ["hello world this is a test", "i love this product",
         "absolutely terrible movie", "fantastic experience"],
        return_tensors="pt", padding=True,
    )
    enc["labels"] = enc["input_ids"].clone()
    opt = torch.optim.AdamW(
        [p for p in m.parameters() if p.requires_grad], lr=1e-3
    )
    losses = []
    for _ in range(15):
        opt.zero_grad()
        out = m(**enc)
        out.loss.backward()
        opt.step()
        losses.append(out.loss.item())
    print(f"  loss: {losses[0]:.3f} -> {losses[-1]:.3f}")
    assert losses[-1] < losses[0]
    print("  [PASS]")


if __name__ == "__main__":
    test_param_count()
    test_training_stochastic()
    test_inference_deterministic()
    test_merge_experts()
    test_mini_training()
    print("\n" + "=" * 60)
    print("[PASS] AdaMix 全部测试通过")
    print("=" * 60)
