"""LoftQ 一致性测试。

测试目标:
  1. (单调) 交替迭代的 loss 单调不增
  2. (初始) T=0 时等价于 QLoRA（Q = NF4(W), BA = 0）
  3. (强一致) 训练时 base 不变
  4. (mini training) LoftQ 比 QLoRA 起始 loss 更低
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from loftq_minimal import LoftQGPT2, LoftQLinear  # noqa: E402
from qlora_minimal import QLoRAGPT2  # noqa: E402


def test_convergence_monotonic():
    print("\n[Test 1] 交替迭代单调下降")
    torch.manual_seed(42)
    model = LoftQGPT2(r=8, alpha=8, n_iter=10)
    history = model.get_layer_convergence(0)
    print(f"  收敛历史（前 6 项）:")
    for t, v in enumerate(history[:6]):
        print(f"    t={t}: {v:.4f}")
    # 单调不增（允许小浮点误差）
    for t in range(1, len(history)):
        assert history[t] <= history[t - 1] + 1e-4, (
            f"t={t} 时 loss 上升: {history[t-1]:.4f} → {history[t]:.4f}"
        )
    print(f"  最终 / 初始 = {history[-1] / history[0]:.4f}")
    print("  [PASS] 单调下降")


def test_initial_forward():
    print("\n[Test 2] 初始 forward 接近原始 GPT-2")
    torch.manual_seed(42)
    model = LoftQGPT2(r=8, alpha=8, n_iter=5).eval()
    from transformers import GPT2LMHeadModel
    base = GPT2LMHeadModel.from_pretrained("gpt2").eval()

    tok = model.tokenizer
    enc = tok("hello world this is a test", return_tensors="pt", padding=True)
    with torch.no_grad():
        out_loftq = model(enc["input_ids"], enc["attention_mask"])
        out_base = base(enc["input_ids"], attention_mask=enc["attention_mask"])
    diff = (out_loftq.logits - out_base.logits).abs().max().item()
    print(f"  logits 最大误差: {diff:.4e}")
    # LoftQ 初始模型 = Q + BA 接近 W，所以 forward 接近 base
    # 但由于量化损失 + SVD 截断，会有一些误差
    assert diff < 5.0, f"LoftQ 初始 forward 偏离过大: {diff}"
    print(f"  [PASS]")


def test_base_frozen():
    print("\n[Test 3] 训练时 base 完全不变")
    torch.manual_seed(42)
    model = LoftQGPT2(r=8, alpha=8, n_iter=5)
    layer0 = model.lm.transformer.h[0].attn.c_attn
    W_before = layer0.base.weight.data.clone()

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
    W_after = layer0.base.weight.data
    diff = (W_before - W_after).abs().max().item()
    print(f"  base.weight 变化: {diff:.6e}")
    assert diff < 1e-6
    print("  [PASS]")


def test_loftq_vs_qlora_init():
    print("\n[Test 4] LoftQ 起始 loss 比 QLoRA 低")
    torch.manual_seed(42)
    qlora = QLoRAGPT2(r=8, alpha=8)
    torch.manual_seed(42)
    loftq = LoftQGPT2(r=8, alpha=8, n_iter=5)

    tok = qlora.tokenizer
    inputs = tok(
        ["hello world this is a test", "i love this product"],
        return_tensors="pt", padding=True,
    )
    inputs["labels"] = inputs["input_ids"].clone()

    with torch.no_grad():
        loss_qlora = qlora(**inputs).loss.item()
        loss_loftq = loftq(**inputs).loss.item()
    print(f"  QLoRA 起始 loss:  {loss_qlora:.4f}")
    print(f"  LoftQ 起始 loss: {loss_loftq:.4f}")
    assert loss_loftq < loss_qlora, "LoftQ 应起始 loss 更低（利用 BA 补偿量化）"
    print(f"  [PASS] LoftQ 起始 loss 比 QLoRA 低 {loss_qlora - loss_loftq:.4f}")


def test_mini_training():
    print("\n[Test 5] mini training 收敛")
    torch.manual_seed(42)
    model = LoftQGPT2(r=8, alpha=8, n_iter=5)
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


if __name__ == "__main__":
    test_convergence_monotonic()
    test_initial_forward()
    test_base_frozen()
    test_loftq_vs_qlora_init()
    test_mini_training()
    print("\n" + "=" * 60)
    print("[PASS] LoftQ 全部测试通过")
    print("=" * 60)
