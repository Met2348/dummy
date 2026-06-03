"""rsLoRA + LoRA+ 扩展的单元测试。

测试目标:
  1. rsLoRA: scaling = α/√r 公式正确
  2. LoRA+: A、B 学习率比例正确
  3. mini training: 两个扩展都能正常收敛
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from lora_extensions import (  # noqa: E402
    RSLoRAGPT2,
    RSLoRALinear,
    lora_plus_param_groups,
)
from lora_minimal import LoRAGPT2  # noqa: E402


def test_rslora_scaling():
    """rsLoRA: scaling = α / sqrt(r)"""
    print("\n[Test 1] rsLoRA scaling 公式 α/√r")
    for r in [4, 8, 16, 64, 256]:
        model = RSLoRAGPT2(r=r, alpha=16)
        for module in model.lm.modules():
            if isinstance(module, RSLoRALinear):
                expected = 16 / math.sqrt(r)
                actual = module.scaling
                assert abs(actual - expected) < 1e-6, (
                    f"r={r}: scaling={actual:.6f}, 期望 {expected:.6f}"
                )
                print(f"  r={r:>4}: scaling = 16/√{r} = {actual:.4f}  [OK]")
                break
        del model
    print("  [PASS]")


def test_lora_plus_param_groups():
    """LoRA+: A、B 不同学习率"""
    print("\n[Test 2] LoRA+ optimizer 参数组")
    torch.manual_seed(42)
    model = LoRAGPT2(r=8, alpha=16)
    groups = lora_plus_param_groups(model, lr_A=1e-4, lambda_B=16.0)

    name_to_lr = {g["_name"]: g["lr"] for g in groups}
    print(f"  lora_A lr = {name_to_lr['lora_A']:.1e}")
    print(f"  lora_B lr = {name_to_lr['lora_B']:.1e}")
    ratio = name_to_lr["lora_B"] / name_to_lr["lora_A"]
    print(f"  比例 = {ratio:.1f} (期望 16.0)")
    assert abs(ratio - 16.0) < 1e-6

    nA = sum(p.numel() for p in [p for g in groups if g["_name"] == "lora_A" for p in g["params"]])
    nB = sum(p.numel() for p in [p for g in groups if g["_name"] == "lora_B" for p in g["params"]])
    print(f"  A 参数: {nA:,}, B 参数: {nB:,}")
    assert nA == 12 * 8 * 768, f"A 参数量 {nA} != 12*8*768"
    assert nB == 12 * 2304 * 8, f"B 参数量 {nB} != 12*2304*8"
    print("  [PASS]")


def test_lora_extensions_mini_training():
    """mini training: LoRA、rsLoRA、LoRA+ 都能正常收敛。"""
    print("\n[Test 3] mini training（15 step on toy 3-sentence batch）")
    torch.manual_seed(42)

    base_model = LoRAGPT2(r=64, alpha=16)
    torch.manual_seed(42)
    rs_model = RSLoRAGPT2(r=64, alpha=16)

    tok = base_model.tokenizer
    inputs = tok(
        ["hello world this is a test sentence",
         "i love this product, highly recommended",
         "absolutely terrible, do not buy"],
        return_tensors="pt", padding=True,
    )
    inputs["labels"] = inputs["input_ids"].clone()

    opt_lora = torch.optim.AdamW(
        [p for p in base_model.parameters() if p.requires_grad], lr=1e-3
    )
    opt_rs = torch.optim.AdamW(
        [p for p in rs_model.parameters() if p.requires_grad], lr=1e-3
    )

    losses_lora, losses_rs = [], []
    for step in range(15):
        for model, opt, losses in [
            (base_model, opt_lora, losses_lora),
            (rs_model, opt_rs, losses_rs),
        ]:
            opt.zero_grad()
            out = model(**inputs)
            out.loss.backward()
            opt.step()
            losses.append(out.loss.item())

    print(f"  LoRA   loss 轨迹: {losses_lora[0]:.3f} → {losses_lora[7]:.3f} → {losses_lora[-1]:.3f}")
    print(f"  rsLoRA loss 轨迹: {losses_rs[0]:.3f} → {losses_rs[7]:.3f} → {losses_rs[-1]:.3f}")
    assert losses_lora[-1] < losses_lora[0], "LoRA loss 没下降"
    assert losses_rs[-1] < losses_rs[0], "rsLoRA loss 没下降"
    print("  [PASS] 两者都正常收敛")


if __name__ == "__main__":
    test_rslora_scaling()
    test_lora_plus_param_groups()
    test_lora_extensions_mini_training()
    print("\n" + "=" * 60)
    print("[PASS] LoRA extensions 单元测试全部通过")
    print("=" * 60)
