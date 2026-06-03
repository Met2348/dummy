"""LoRA 最小实现（手写）。

对应论文：Hu et al., 2021, arXiv:2106.09685
对应 lecture: lectures/01-lora.md

核心思想：
  - 冻结预训练 W_0
  - 在每个 target Linear/Conv1D 上加并行的低秩分支 BA
  - 训练目标只更新 A, B（公式 1, 2）
  - 训练后可合并: W' = W_0 + α/r BA（merge_weights）

公式索引（来自 lecture）:
    h = W_0 x + (α/r) B A x       (1)
    B ← 0,  A ~ N(0, σ²)           (2)
    |φ_layer| = 2 r d              (3)
    Δ W(r=d) ≡ Full-FT             (4)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.pytorch_utils import Conv1D

sys.path.append(str(Path(__file__).parent))
from common import (  # noqa: E402
    freeze_base_model,
    get_in_out_dims,
    get_parent_and_attr,
    is_conv1d,
    print_param_summary,
    target_linear_modules,
)


class LoRALinear(nn.Module):
    """单层 LoRA: h = base(x) + α/r * B A x。

    兼容 nn.Linear 和 GPT-2 的 Conv1D（两者 weight 形状不同）。
    """

    def __init__(self, base_linear: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base_linear
        # 冻结预训练权重（公式 5 的 W_0）
        for p in self.base.parameters():
            p.requires_grad = False

        d_in, d_out = get_in_out_dims(base_linear)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r  # 公式 (1) 的 α/r

        # 公式 (2): A 用 Kaiming 初始化, B 用零初始化
        self.A = nn.Parameter(torch.empty(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.is_conv1d = is_conv1d(base_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """公式 (1) 的前向。"""
        base_out = self.base(x)
        # LoRA 分支: x @ A.T @ B.T，与 nn.Linear semantics 一致
        # (B, n, d_in) @ (d_in, r) @ (r, d_out) = (B, n, d_out)
        lora_out = self.dropout(x) @ self.A.T @ self.B.T
        return base_out + self.scaling * lora_out

    @torch.no_grad()
    def merge_weights(self) -> None:
        """把 LoRA 合并回 base.weight，删除 A、B。

        合并后 LoRALinear 等价于一个被更新过权重的 base_linear。
        """
        delta = self.scaling * (self.B @ self.A)  # (d_out, d_in)
        if self.is_conv1d:
            # Conv1D.weight shape (in, out)，需要转置
            self.base.weight.data.add_(delta.T)
        else:
            self.base.weight.data.add_(delta)


class LoRAGPT2(nn.Module):
    """把 GPT-2 base 包装成 LoRA 模型。

    默认给每层的 `c_attn`（合并的 q/k/v Conv1D）打 LoRA。
    """

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

        # 公式 5: θ_LM 全部冻结
        freeze_base_model(self.lm)

        # 替换每个 target 模块
        matches = target_linear_modules(self.lm, target_modules)
        for qname, old in matches:
            parent, attr = get_parent_and_attr(self.lm, qname)
            new = LoRALinear(old, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, attr, new)

        self.r = r
        self.alpha = alpha
        self.target_modules = target_modules

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def merge_all(self) -> None:
        """把所有 LoRA 合并回 base，使模型变成普通微调后的样子。"""
        for m in self.lm.modules():
            if isinstance(m, LoRALinear):
                m.merge_weights()


def main() -> None:
    torch.manual_seed(42)
    model = LoRAGPT2(r=8, alpha=16)
    print_param_summary(model, "LoRAGPT2 (r=8, target=c_attn)")
    # Expected trainable: 12 层 × (8 * 768 + 2304 * 8) = 12 × (6144 + 18432) = 12 × 24576 = 294,912

    print(f"\n参数布局：")
    print(f"  每层 c_attn LoRA:")
    print(f"    A shape: (r=8, d_in=768)        = 6,144")
    print(f"    B shape: (d_out=2304, r=8)      = 18,432")
    print(f"    subtotal per layer:               24,576")
    print(f"  12 层合计:                       294,912")

    enc = model.tokenizer("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"])
    print(f"\n前向输出 logits.shape={tuple(out.logits.shape)}")


if __name__ == "__main__":
    main()
