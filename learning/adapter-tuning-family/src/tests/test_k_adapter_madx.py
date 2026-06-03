"""K-Adapter + MAD-X 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from k_adapter_minimal import MultiKnowledgeGPT2, TOY_FACTUAL_TRIPLES  # noqa: E402
from madx_minimal import MADXGPT2, TOY_MULTILINGUAL_DATA  # noqa: E402


def test_k_adapter_param():
    print("\n[Test 1] K-Adapter 参数量 = N 类 × 12 × per_layer")
    torch.manual_seed(42)
    m = MultiKnowledgeGPT2(r=16, knowledge_types=("factual", "linguistic"))
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    expected = 2 * 12 * (2 * 768 * 16 + 16 + 768)
    print(f"  trainable: {n:,}  expected: {expected:,}")
    assert n == expected
    print("  [PASS]")


def test_k_adapter_freeze():
    print("\n[Test 2] K-Adapter 冻结指定类别")
    torch.manual_seed(42)
    m = MultiKnowledgeGPT2(r=16, knowledge_types=("factual", "linguistic"))
    n0 = sum(p.numel() for p in m.parameters() if p.requires_grad)
    m.freeze_adapter("factual")
    n1 = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  冻结前: {n0:,}")
    print(f"  冻结 factual 后: {n1:,}")
    assert n1 == n0 // 2  # 一半被冻结
    print("  [PASS]")


def test_madx_param():
    print("\n[Test 3] MAD-X 参数量 = LA × N + TA × M + IA × N")
    torch.manual_seed(42)
    m = MADXGPT2(r=16, languages=("en", "de", "fr"), tasks=("ner",))
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    per_adapter = 2 * 768 * 16 + 16 + 768
    expected = (3 + 1) * 12 * per_adapter + 3 * 2 * 768  # LA+TA + IA
    print(f"  trainable: {n:,}  expected: {expected:,}")
    assert n == expected
    print("  [PASS]")


def test_madx_lang_switch():
    print("\n[Test 4] MAD-X 切换语言（训练后看差异）")
    torch.manual_seed(42)
    m = MADXGPT2(r=16, languages=("en", "de", "fr"), tasks=("ner",))
    tok = m.tokenizer

    # 先随机扰动各 lang adapter 的 up 权重（模拟"训过"）
    with torch.no_grad():
        for block in m.lm.transformer.h:
            for lang in ("en", "de", "fr"):
                la = block.mlp.language_adapters[lang]
                la.up.weight.normal_(std=0.01)

    enc_en = tok("hello world", return_tensors="pt", padding=True)
    m.set_active("en", "ner")
    m.eval()
    with torch.no_grad():
        out_en = m(enc_en["input_ids"], enc_en["attention_mask"]).logits

    m.set_active("de", "ner")
    with torch.no_grad():
        out_de = m(enc_en["input_ids"], enc_en["attention_mask"]).logits

    # 现在不同 lang adapter -> 不同输出
    diff = (out_en - out_de).abs().max().item()
    print(f"  en vs de 同输入下 logits 差异: {diff:.4e}")
    assert diff > 1e-3, "训练后应该不同"
    print("  [PASS]")


def test_mini_training():
    print("\n[Test 5] K-Adapter mini training")
    torch.manual_seed(42)
    m = MultiKnowledgeGPT2(r=16, knowledge_types=("factual",))
    tok = m.tokenizer
    enc = tok(TOY_FACTUAL_TRIPLES[:4], return_tensors="pt", padding=True)
    enc["labels"] = enc["input_ids"].clone()
    opt = torch.optim.AdamW(
        [p for p in m.parameters() if p.requires_grad], lr=1e-3
    )
    losses = []
    for _ in range(10):
        opt.zero_grad()
        out = m(**enc)
        out.loss.backward()
        opt.step()
        losses.append(out.loss.item())
    print(f"  loss: {losses[0]:.3f} -> {losses[-1]:.3f}")
    assert losses[-1] < losses[0]
    print("  [PASS]")


if __name__ == "__main__":
    test_k_adapter_param()
    test_k_adapter_freeze()
    test_madx_param()
    test_madx_lang_switch()
    test_mini_training()
    print("\n" + "=" * 60)
    print("[PASS] K-Adapter + MAD-X 全部测试通过")
    print("=" * 60)
