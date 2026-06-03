"""AdapterDrop + Compacter 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from adapterdrop_minimal import AdapterDropGPT2  # noqa: E402
from compacter_minimal import CompacterGPT2, PHMLinear, kronecker  # noqa: E402


def test_kronecker_correctness():
    print("\n[Test 1] Kronecker 积正确性")
    # 2x2 ⊗ 2x2 -> 4x4 数值验证
    A = torch.tensor([[1., 2.], [3., 4.]])
    B = torch.tensor([[0., 5.], [6., 7.]])
    expected = torch.tensor([
        [0., 5., 0., 10.],
        [6., 7., 12., 14.],
        [0., 15., 0., 20.],
        [18., 21., 24., 28.],
    ])
    got = kronecker(A, B)
    print(f"  got:\n{got}")
    print(f"  expected:\n{expected}")
    assert torch.allclose(got, expected), "Kronecker 错误"
    print("  [PASS]")


def test_phm_param_count():
    print("\n[Test 2] PHMLinear 参数量")
    layer = PHMLinear(in_features=16, out_features=768, n=4)
    n_total = sum(p.numel() for p in layer.parameters())
    # A: 4*4*4 = 64
    # B: 4 * 192 * 4 = 3072
    # bias: 768
    expected = 64 + 3072 + 768
    print(f"  got: {n_total:,}  expected: {expected:,}")
    assert n_total == expected
    # 与普通 Linear 对比
    normal = torch.nn.Linear(16, 768)
    n_normal = sum(p.numel() for p in normal.parameters())
    print(f"  Normal Linear: {n_normal:,}")
    print(f"  压缩比: {n_normal / n_total:.2f}x")
    print("  [PASS]")


def test_compacter_total():
    print("\n[Test 3] CompacterGPT2 总参数量 < Pfeiffer")
    torch.manual_seed(42)
    model = CompacterGPT2(r=16, n=4)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Compacter trainable: {n_train:,}")
    print(f"  Pfeiffer baseline:   304,320")
    print(f"  压缩比: {304320 / n_train:.2f}x")
    assert n_train < 304320  # 必须比 Pfeiffer 小
    print("  [PASS]")


def test_compacter_initial_forward():
    print("\n[Test 4] Compacter 初始 forward = base")
    from transformers import GPT2LMHeadModel
    torch.manual_seed(42)
    model = CompacterGPT2(r=16, n=4).eval()
    base = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    enc = model.tokenizer("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out_c = model(enc["input_ids"], enc["attention_mask"])
        out_b = base(enc["input_ids"], attention_mask=enc["attention_mask"])
    diff = (out_c.logits - out_b.logits).abs().max().item()
    print(f"  max |Δlogits|: {diff:.4e}")
    assert diff < 1e-5
    print("  [PASS]")


def test_compacter_mini_training():
    print("\n[Test 5] Compacter mini training")
    torch.manual_seed(42)
    model = CompacterGPT2(r=16, n=4)
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


def test_adapterdrop_inference():
    print("\n[Test 6] AdapterDrop 推理时永久丢")
    torch.manual_seed(42)
    model = AdapterDropGPT2(r=16).eval()
    assert model.get_active_layers() == 12
    model.set_inference_drop(k=5)
    assert model.get_active_layers() == 7
    model.set_inference_drop(k=11)
    assert model.get_active_layers() == 1
    model.set_inference_drop(k=0)
    assert model.get_active_layers() == 12
    print("  [PASS]")


if __name__ == "__main__":
    test_kronecker_correctness()
    test_phm_param_count()
    test_compacter_total()
    test_compacter_initial_forward()
    test_compacter_mini_training()
    test_adapterdrop_inference()
    print("\n" + "=" * 60)
    print("[PASS] AdapterDrop + Compacter 全部测试通过")
    print("=" * 60)
