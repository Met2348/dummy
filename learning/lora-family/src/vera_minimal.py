"""VeRA 最小实现（手写）。

对应论文：Kopiczko et al. 2024, arXiv:2310.11454
对应 lecture: lectures/04-vera.md

核心：
  - A, B 冻结为随机矩阵（Kaiming uniform，固定 seed）
  - A, B 跨所有层共享（一对，不是每层一对）
  - Λ_b ∈ R^r, Λ_d ∈ R^d 是每层独立的可训练对角向量
  - forward: h = W_0 x + α/r · Λ_d ⊙ (B (Λ_b ⊙ (A x)))

参数量：L × (r + d) ≈ L × d，比 LoRA 少 2r 倍
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


class VeRASharedBuffers:
    """全局共享的 A, B 随机矩阵（固定 seed）。

    所有 VeRALinear 实例从这里读取 A、B。
    """
    _A: torch.Tensor | None = None
    _B: torch.Tensor | None = None
    _seed: int | None = None

    @classmethod
    def init(cls, d_max_in: int, d_max_out: int, r: int, seed: int = 42) -> None:
        """初始化共享 A、B。

        d_max_in/d_max_out: 模型中最大的输入/输出维度（用于一次性分配，按需切片）
        """
        g = torch.Generator().manual_seed(seed)
        bound = math.sqrt(6.0 / r)
        cls._A = torch.empty(r, d_max_in).uniform_(-bound, bound, generator=g)
        cls._B = torch.empty(d_max_out, r).uniform_(-bound, bound, generator=g)
        cls._seed = seed

    @classmethod
    def get_A(cls, r: int, d_in: int) -> torch.Tensor:
        assert cls._A is not None, "VeRASharedBuffers.init() 必须先调用"
        return cls._A[:r, :d_in]

    @classmethod
    def get_B(cls, r: int, d_out: int) -> torch.Tensor:
        assert cls._B is not None, "VeRASharedBuffers.init() 必须先调用"
        return cls._B[:d_out, :r]

    @classmethod
    def to(cls, device, dtype=None):
        if cls._A is not None:
            cls._A = cls._A.to(device=device, dtype=dtype or cls._A.dtype)
        if cls._B is not None:
            cls._B = cls._B.to(device=device, dtype=dtype or cls._B.dtype)


class VeRALinear(nn.Module):
    """单层 VeRA。"""

    def __init__(
        self,
        base_linear: nn.Module,
        r: int = 256,
        alpha: int | None = None,
        d_initial: float = 0.1,
    ):
        super().__init__()
        for p in base_linear.parameters():
            p.requires_grad = False
        self.base = base_linear

        d_in, d_out = get_in_out_dims(base_linear)
        self.r = r
        self.d_in = d_in
        self.d_out = d_out
        self.scaling = (alpha if alpha is not None else r) / r

        # 公式 (3): Λ_b = 1, Λ_d = d_initial
        self.Lambda_b = nn.Parameter(torch.ones(r))
        self.Lambda_d = nn.Parameter(torch.full((d_out,), d_initial))
        self.is_conv1d = is_conv1d(base_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """公式 (1) 前向。"""
        A = VeRASharedBuffers.get_A(self.r, self.d_in).to(x.device, x.dtype)
        B = VeRASharedBuffers.get_B(self.r, self.d_out).to(x.device, x.dtype)
        # x @ A.T → (..., r)
        # ⊙ Λ_b → (..., r)
        # @ B.T → (..., d_out)
        # ⊙ Λ_d → (..., d_out)
        out = (x @ A.T) * self.Lambda_b
        out = (out @ B.T) * self.Lambda_d
        return self.base(x) + self.scaling * out


class VeRAGPT2(nn.Module):
    """GPT-2 + VeRA。"""

    def __init__(
        self,
        base_model_name: str = "gpt2",
        r: int = 256,
        alpha: int | None = None,
        d_initial: float = 0.1,
        target_modules: tuple[str, ...] = ("c_attn",),
        seed: int = 42,
    ):
        super().__init__()
        self.lm = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        freeze_base_model(self.lm)

        # 找到所有 target 模块，确定 d_max_in/out
        matches = target_linear_modules(self.lm, target_modules)
        d_max_in, d_max_out = 0, 0
        for _, m in matches:
            d_in, d_out = get_in_out_dims(m)
            d_max_in = max(d_max_in, d_in)
            d_max_out = max(d_max_out, d_out)

        # 初始化全局共享 A, B
        VeRASharedBuffers.init(d_max_in=d_max_in, d_max_out=d_max_out, r=r, seed=seed)

        # 替换每个 target
        for qname, old in matches:
            parent, attr = get_parent_and_attr(self.lm, qname)
            new = VeRALinear(old, r=r, alpha=alpha, d_initial=d_initial)
            setattr(parent, attr, new)

        self.r = r
        self.alpha = alpha if alpha is not None else r

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def to(self, *args, **kwargs):
        """在 to(device/dtype) 时也把共享 buffer 搬过去。"""
        out = super().to(*args, **kwargs)
        VeRASharedBuffers.to(*args, **kwargs)
        return out


def main() -> None:
    torch.manual_seed(42)
    model = VeRAGPT2(r=256, alpha=256, d_initial=0.1)
    print_param_summary(model, "VeRA (r=256, target=c_attn)")
    # Expected: 12 层 × (256 + 2304) = 12 × 2560 = 30,720 trainable
    print(f"\n参数布局（per layer）：")
    print(f"  Λ_b shape: (r=256,)         = 256")
    print(f"  Λ_d shape: (d_out=2304,)    = 2,304")
    print(f"  subtotal per layer:         = 2,560")
    print(f"  12 层合计:                   = 30,720")
    print(f"\n共享 A、B（不计入 trainable，所有层共用）:")
    print(f"  A shape: (r=256, d_in=768)  = 196,608  (frozen)")
    print(f"  B shape: (d_out=2304, r=256) = 589,824  (frozen)")

    # 对比 LoRA r=8: c_attn 是合并 q/k/v 的 Conv1D, d_in=768, d_out=2304
    lora_params = 12 * (8 * 768 + 2304 * 8)  # A=(8,768) + B=(2304,8) per layer
    print(f"\n对比 LoRA r=8: {lora_params:,} trainable")
    vera_params = 12 * (256 + 2304)
    print(f"VeRA r=256:    {vera_params:,} trainable")
    print(f"VeRA / LoRA = {vera_params / lora_params:.2%}（VeRA 节省 {lora_params / vera_params:.1f}x）")

    enc = model.tokenizer("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"])
    print(f"\n前向输出 logits.shape={tuple(out.logits.shape)}")


if __name__ == "__main__":
    main()
