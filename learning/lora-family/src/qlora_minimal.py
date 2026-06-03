"""QLoRA 最小实现（fake-quant 主战场）。

对应论文：Dettmers et al. 2023, arXiv:2305.14314
对应 lecture: lectures/06-qlora.md

核心：base.weight 用 NF4 fake-quant 替换（fake-quant，CPU/GPU 都跑），
LoRA 部分与标准 LoRA 完全一致。

公式：
    h = dequant(NF4(W_0)) · x + α/r BA · x
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent))
from common import (  # noqa: E402
    freeze_base_model,
    get_in_out_dims,
    get_parent_and_attr,
    is_conv1d,
    print_param_summary,
    target_linear_modules,
)
from nf4_quant import nf4_quant_dequant  # noqa: E402


class QLoRALinear(nn.Module):
    """单层 QLoRA: NF4 base + LoRA(A, B)。"""

    def __init__(
        self,
        base_linear: nn.Module,
        r: int = 8,
        alpha: int = 16,
        block_size: int = 64,
    ):
        super().__init__()
        d_in, d_out = get_in_out_dims(base_linear)
        self.is_conv1d = is_conv1d(base_linear)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.block_size = block_size

        # 1. 用 NF4 fake-quant 替换 base.weight
        with torch.no_grad():
            W_quantized = nf4_quant_dequant(base_linear.weight.data, block_size=block_size)
            base_linear.weight.data.copy_(W_quantized)
        for p in base_linear.parameters():
            p.requires_grad = False
        self.base = base_linear

        # 2. LoRA: B 零初始化（公式 2 同 LoRA）
        self.A = nn.Parameter(torch.empty(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """h = base_quantized(x) + α/r BA x"""
        base_out = self.base(x)
        lora_out = x @ self.A.T @ self.B.T
        return base_out + self.scaling * lora_out

    @torch.no_grad()
    def quantization_error(self, original_weight: torch.Tensor) -> float:
        """返回 base.weight 与原始 W_0 的 RMSE，验证量化损失。"""
        if self.is_conv1d:
            diff = (self.base.weight.data - original_weight)
        else:
            diff = (self.base.weight.data - original_weight)
        return diff.pow(2).mean().sqrt().item()


class QLoRAGPT2(nn.Module):
    """GPT-2 + QLoRA。"""

    def __init__(
        self,
        base_model_name: str = "gpt2",
        r: int = 8,
        alpha: int = 16,
        block_size: int = 64,
        target_modules: tuple[str, ...] = ("c_attn",),
    ):
        super().__init__()
        self.lm = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        freeze_base_model(self.lm)

        matches = target_linear_modules(self.lm, target_modules)
        for qname, old in matches:
            parent, attr = get_parent_and_attr(self.lm, qname)
            new = QLoRALinear(old, r=r, alpha=alpha, block_size=block_size)
            setattr(parent, attr, new)

        self.r = r
        self.alpha = alpha
        self.block_size = block_size

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )


def main() -> None:
    torch.manual_seed(42)

    # 加载原始 GPT-2 看 W_0
    base_orig = GPT2LMHeadModel.from_pretrained("gpt2")
    W_0_layer0 = base_orig.transformer.h[0].attn.c_attn.weight.data.clone()

    # 构造 QLoRA
    model = QLoRAGPT2(r=8, alpha=16, block_size=64)
    print_param_summary(model, "QLoRA (r=8, NF4 fake-quant)")

    # 量化误差
    qlora_layer0 = model.lm.transformer.h[0].attn.c_attn
    err = qlora_layer0.quantization_error(W_0_layer0)
    rel_err = err / W_0_layer0.std().item()
    print(f"\nNF4 量化 base.weight 的 RMSE: {err:.5f}")
    print(f"相对 RMSE (vs std): {rel_err:.4f}")

    enc = model.tokenizer("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"])
    print(f"\nforward logits.shape={tuple(out.logits.shape)}")

    # 训练几步看 base 不变
    inputs = model.tokenizer(["hello world"], return_tensors="pt", padding=True)
    inputs["labels"] = inputs["input_ids"].clone()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    W_before = qlora_layer0.base.weight.data.clone()
    for _ in range(5):
        optimizer.zero_grad()
        out = model(**inputs)
        out.loss.backward()
        optimizer.step()
    W_after = qlora_layer0.base.weight.data
    print(f"\n训练 5 step 后 base.weight 是否变化: {(W_before - W_after).abs().max().item():.6e}")
    print(f"  (应为 0，base 不更新)")


if __name__ == "__main__":
    main()
