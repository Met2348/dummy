"""LoKr 最小实现（手写）。

对应论文：LyCORIS, Yeh et al. 2023, arXiv:2309.14859
对应 lecture: lectures/05-loha-lokr.md

核心：ΔW = B ⊗ A（Kronecker 积）
  - d_out = m_1 * m_2, d_in = n_1 * n_2
  - A ∈ R^(m_1, n_1), B ∈ R^(m_2, n_2)
  - 额外低秩约束：A = B_lr @ A_lr，进一步压参
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


def _find_factor(d: int, target_factor: int) -> tuple[int, int]:
    """找到 d = m1 * m2 的分解，m1 接近 target_factor 且整除。"""
    if d % target_factor == 0:
        return target_factor, d // target_factor
    # 找最接近的因子
    best = (1, d)
    best_diff = abs(1 - target_factor)
    for m1 in range(1, int(d**0.5) + 1):
        if d % m1 == 0:
            for cand in [m1, d // m1]:
                diff = abs(cand - target_factor)
                if diff < best_diff:
                    best = (cand, d // cand)
                    best_diff = diff
    return best


class LoKrLinear(nn.Module):
    """单层 LoKr: ΔW = B ⊗ (B_lr @ A_lr)。

    分解：
      A_lr (r, n_1), B_lr (m_1, r) → A = B_lr @ A_lr ∈ R^(m_1, n_1)
      B ∈ R^(m_2, n_2)
      ΔW = B ⊗ A ∈ R^(m_1 m_2, n_1 n_2) = R^(d_out, d_in)
    """

    def __init__(
        self,
        base_linear: nn.Module,
        factor: int = 8,
        r: int = 4,
        alpha: int = 4,
    ):
        super().__init__()
        for p in base_linear.parameters():
            p.requires_grad = False
        d_in, d_out = get_in_out_dims(base_linear)

        # 分解维度
        m1, m2 = _find_factor(d_out, factor)
        n1, n2 = _find_factor(d_in, factor)

        # 公式 (3): A 小（用 rank-r 约束），B 小
        self.A_lr = nn.Parameter(torch.empty(r, n1))
        self.B_lr = nn.Parameter(torch.zeros(m1, r))  # 零初始化让初始 A=0
        self.B = nn.Parameter(torch.empty(m2, n2))
        nn.init.kaiming_uniform_(self.A_lr, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

        self.base = base_linear
        self.m1, self.m2, self.n1, self.n2 = m1, m2, n1, n2
        self.r = r
        self.scaling = alpha / r
        self.is_conv1d = is_conv1d(base_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = self.B_lr @ self.A_lr           # (m1, n1)
        delta = torch.kron(self.B, A)        # (m1*m2, n1*n2) = (d_out, d_in)
        return self.base(x) + self.scaling * (x @ delta.T)

    @torch.no_grad()
    def get_delta_W(self) -> torch.Tensor:
        A = self.B_lr @ self.A_lr
        return self.scaling * torch.kron(self.B, A)


class LoKrGPT2(nn.Module):
    """GPT-2 + LoKr。"""

    def __init__(
        self,
        base_model_name: str = "gpt2",
        factor: int = 8,
        r: int = 4,
        alpha: int = 4,
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
            new = LoKrLinear(old, factor=factor, r=r, alpha=alpha)
            setattr(parent, attr, new)

        self.factor = factor
        self.r = r

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )


def main() -> None:
    torch.manual_seed(42)
    # factor 应整除 d_in=768 和 d_out=2304。32 同时整除两者
    # 在 d_in=768 上：n1=32, n2=24
    # 在 d_out=2304 上：m1=32, m2=72
    model = LoKrGPT2(factor=32, r=4)
    print_param_summary(model, "LoKr (factor=32, r=4)")

    # 检查 layer 0 分解
    layer0 = model.lm.transformer.h[0].attn.c_attn
    print(f"\nlayer 0 分解:")
    print(f"  d_in=768 → m1*n1 形式: ({layer0.m1}, {layer0.n1}) for A")
    print(f"  d_out=2304 → m2*n2 形式: ({layer0.m2}, {layer0.n2}) for B")
    print(f"  A_lr params: {layer0.r} × {layer0.n1} = {layer0.r * layer0.n1}")
    print(f"  B_lr params: {layer0.m1} × {layer0.r} = {layer0.m1 * layer0.r}")
    print(f"  B params:    {layer0.m2} × {layer0.n2} = {layer0.m2 * layer0.n2}")

    enc = model.tokenizer("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"])
    print(f"\nforward logits.shape={tuple(out.logits.shape)}")


if __name__ == "__main__":
    main()
