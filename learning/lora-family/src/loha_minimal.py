"""LoHa 最小实现（手写）。

对应论文：FedPara, Hyeon-Woo et al. 2022, arXiv:2108.06098
对应 lecture: lectures/05-loha-lokr.md

核心：ΔW = (B_1 A_1) ⊙ (B_2 A_2)，Hadamard 积让等效秩为 r²（vs LoRA 的 r）。
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


class LoHaLinear(nn.Module):
    """单层 LoHa: h = base(x) + α/r · ((B_1 A_1) ⊙ (B_2 A_2)) x"""

    def __init__(self, base_linear: nn.Module, r: int = 8, alpha: int = 16):
        super().__init__()
        for p in base_linear.parameters():
            p.requires_grad = False
        d_in, d_out = get_in_out_dims(base_linear)

        # 公式 (1): 两对低秩因子
        self.A_1 = nn.Parameter(torch.empty(r, d_in))
        self.B_1 = nn.Parameter(torch.zeros(d_out, r))  # 零初始化保证初始 ΔW=0
        self.A_2 = nn.Parameter(torch.empty(r, d_in))
        self.B_2 = nn.Parameter(torch.empty(d_out, r))
        nn.init.kaiming_uniform_(self.A_1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.A_2, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B_2, a=math.sqrt(5))

        self.base = base_linear
        self.r = r
        self.scaling = alpha / r
        self.is_conv1d = is_conv1d(base_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 直接构造 ΔW 然后 x @ ΔW^T。效率不佳但易读
        delta_1 = self.B_1 @ self.A_1  # (d_out, d_in), rank ≤ r
        delta_2 = self.B_2 @ self.A_2
        delta = delta_1 * delta_2       # Hadamard, rank ≤ r²
        return self.base(x) + self.scaling * (x @ delta.T)

    @torch.no_grad()
    def get_delta_W(self) -> torch.Tensor:
        delta_1 = self.B_1 @ self.A_1
        delta_2 = self.B_2 @ self.A_2
        return self.scaling * (delta_1 * delta_2)


class LoHaGPT2(nn.Module):
    """GPT-2 + LoHa。"""

    def __init__(
        self,
        base_model_name: str = "gpt2",
        r: int = 8,
        alpha: int = 16,
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
            new = LoHaLinear(old, r=r, alpha=alpha)
            setattr(parent, attr, new)

        self.r = r
        self.alpha = alpha

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )


def main() -> None:
    torch.manual_seed(42)
    model = LoHaGPT2(r=8, alpha=16)
    print_param_summary(model, "LoHa (r=8)")
    # Expected: 12 × 4 × 8 × (768+2304) ?
    # 实际: 12 × (8*768 + 2304*8 + 8*768 + 2304*8) = 12 × 49152 = 589,824
    print(f"\n参数布局（per layer）：")
    print(f"  A_1, B_1: 2 × 8 × (768 + 2304) = {2 * 8 * (768 + 2304):,}")
    print(f"  A_2, B_2: 同上 = {2 * 8 * (768 + 2304):,}")
    print(f"  total per layer: {4 * 8 * (768 + 2304):,}")
    print(f"  12 层合计: {12 * 4 * 8 * (768 + 2304):,}")

    enc = model.tokenizer("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"])
    print(f"\nforward logits.shape={tuple(out.logits.shape)}")


if __name__ == "__main__":
    main()
