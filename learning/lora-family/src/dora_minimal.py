"""DoRA 最小实现（手写）。

对应论文：Liu et al. 2024, arXiv:2402.09353
对应 lecture: lectures/08-dora.md

核心：
  W = m · V / ||V||_c
  V = W_0 + BA
  forward: h = m · (W_0 + BA) / ||W_0 + BA||_c · x
  反向：用 detach(||V||_c) 避免二阶导

公式：
    (1) W = m · V / ||V||_c
    (2) V = W_0 + α/r BA
    (3) ∂forward / ∂params 用 detach 简化
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


def _extract_weight_out_in(base_linear: nn.Module) -> torch.Tensor:
    """统一返回 (d_out, d_in) 的权重张量。"""
    if is_conv1d(base_linear):
        return base_linear.weight.data.T.clone()  # Conv1D weight is (in, out)
    return base_linear.weight.data.clone()  # nn.Linear weight is (out, in)


class DoRALinear(nn.Module):
    """单层 DoRA: h = m · (W_0 + α/r BA) / ||W_0 + α/r BA||_c · x"""

    def __init__(
        self,
        base_linear: nn.Module,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        for p in base_linear.parameters():
            p.requires_grad = False

        d_in, d_out = get_in_out_dims(base_linear)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.is_conv1d = is_conv1d(base_linear)

        # 公式 (1): 提取 W_0 (d_out, d_in)，计算 column-wise norm
        W_0 = _extract_weight_out_in(base_linear)
        # column-wise norm = 沿 d_out 维度求 norm，每"列"（d_in 方向）一个标量
        m_init = W_0.norm(dim=0)  # (d_in,)
        self.m = nn.Parameter(m_init.clone())

        # LoRA
        self.A = nn.Parameter(torch.empty(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))  # 公式 (2): B 零初始化
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.base = base_linear

    def _compute_W_dora(self, x_for_dropout: torch.Tensor | None = None) -> torch.Tensor:
        """返回当前 W_DoRA (d_out, d_in)。x_for_dropout 仅用于 dropout 路径。"""
        W_0 = _extract_weight_out_in(self.base)  # (d_out, d_in)
        # ΔW = α/r BA, shape (d_out, d_in)
        delta = self.scaling * (self.B @ self.A)
        V = W_0 + delta
        # 公式 (3): detach norm 避免二阶导
        norm = V.norm(dim=0).clamp(min=1e-8).detach()  # (d_in,)
        # broadcast: W_dora[i, j] = m[j] / norm[j] * V[i, j]
        scale = (self.m / norm).unsqueeze(0)  # (1, d_in)
        return scale * V  # (d_out, d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_dora = self._compute_W_dora()
        # forward semantics: 对 nn.Linear 是 x @ W^T + b
        # 对 Conv1D 是 x @ W + b，但我们的 W_dora 是 (out, in)
        # 统一用 x @ W^T 形式
        out = x @ W_dora.T
        if self.base.bias is not None:
            out = out + self.base.bias
        return out

    @torch.no_grad()
    def merge_to_base(self) -> None:
        """把 W_dora 合并回 base.weight。"""
        W_dora = self._compute_W_dora()
        if self.is_conv1d:
            self.base.weight.data.copy_(W_dora.T)
        else:
            self.base.weight.data.copy_(W_dora)


class DoRAGPT2(nn.Module):
    """GPT-2 + DoRA。"""

    def __init__(
        self,
        base_model_name: str = "gpt2",
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
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
            new = DoRALinear(old, r=r, alpha=alpha, dropout=dropout)
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
    model = DoRAGPT2(r=8, alpha=16)
    print_param_summary(model, "DoRA (r=8)")
    # Expected: 12 × (m=768 + A=8*768 + B=2304*8)
    #         = 12 × (768 + 6144 + 18432)
    #         = 12 × 25344 = 304,128
    print(f"\n参数布局 per layer:")
    print(f"  m  shape: (d_in=768,)      = 768")
    print(f"  A  shape: (r=8, 768)       = 6,144")
    print(f"  B  shape: (2304, r=8)      = 18,432")
    print(f"  per layer total           = 25,344")
    print(f"  12 层合计                 = 304,128")
    print(f"\n比 LoRA 多: {12 * 768:,} = magnitude 参数")

    # 验证初始 W = W_0
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    layer0 = model.lm.transformer.h[0].attn.c_attn
    W_dora = layer0._compute_W_dora()
    W_0 = _extract_weight_out_in(base.transformer.h[0].attn.c_attn)
    diff = (W_dora - W_0).abs().max().item()
    print(f"\n初始 W_DoRA vs W_0 最大误差: {diff:.4e}")
    print(f"  → 公式 (1) 验证: m * V/||V||_c = W_0 (当 BA=0 时)")

    enc = model.tokenizer("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"])
    print(f"\nforward logits.shape={tuple(out.logits.shape)}")


if __name__ == "__main__":
    main()
