"""AdaLoRA 最小实现（手写）。

对应论文：Zhang et al. 2023, arXiv:2303.10512
对应 lecture: lectures/02-adalora.md

核心：把 LoRA 的 ΔW = BA 改成 SVD 形式 ΔW = P Λ Q^T，
训练时计算重要性打分 S_i = |λ_i · ∂L/∂λ_i| (EMA)，
周期性剪掉低打分的 singular value。

公式索引：
    ΔW = P Λ Q^T                                     (1)
    R(P, Q) = ||P^T P - I||² + ||Q^T Q - I||²        (2)
    S_i = |λ_i · ∇λ_i L|                              (3)
    r(t) = r_f + (r_i - r_f)(1 - (t-t_w)/(T-t_w))³  (4)
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


class AdaLoRALinear(nn.Module):
    """单层 AdaLoRA: h = base(x) + α/r * P diag(Λ) Q^T x。

    参数布局：
        P: (d_out, r_init)  ← 左奇异向量
        Λ: (r_init,)        ← 对角奇异值（初始为零，同 LoRA B）
        Q: (d_in, r_init)   ← 右奇异向量（存 Q，forward 时用 x @ Q）
    """

    def __init__(
        self,
        base_linear: nn.Module,
        r_init: int = 12,
        alpha: int = 16,
        ortho_lambda: float = 0.1,
    ):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        d_in, d_out = get_in_out_dims(base_linear)
        self.r_init = r_init
        self.alpha = alpha
        self.scaling = alpha / r_init
        self.ortho_lambda = ortho_lambda
        self.is_conv1d = is_conv1d(base_linear)

        # 公式 (1): SVD 形式 P, Λ, Q
        self.P = nn.Parameter(torch.empty(d_out, r_init))
        self.Lambda = nn.Parameter(torch.zeros(r_init))     # 公式 (2) 零初始化
        self.Q = nn.Parameter(torch.empty(d_in, r_init))
        nn.init.normal_(self.P, std=0.02)
        nn.init.normal_(self.Q, std=0.02)

        # 重要性打分（EMA buffer，不参与梯度）
        self.register_buffer("S_ema", torch.zeros(r_init))
        # 剪枝 mask（初始全 1）
        self.register_buffer("mask", torch.ones(r_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：h = base(x) + α/r * (x @ Q) ⊙ (Λ * mask) @ P^T"""
        base_out = self.base(x)
        masked_lambda = self.Lambda * self.mask  # masked diag
        # x @ Q → (..., r)
        # 与 (Λ * mask) element-wise → (..., r)
        # @ P^T → (..., d_out)
        lora_out = (x @ self.Q) * masked_lambda @ self.P.T
        return base_out + self.scaling * lora_out

    def ortho_loss(self) -> torch.Tensor:
        """公式 (2) 的正交正则项。"""
        r = self.P.shape[1]
        I_r = torch.eye(r, device=self.P.device, dtype=self.P.dtype)
        # P^T P 是 (r, r)
        loss_P = ((self.P.T @ self.P - I_r) ** 2).sum()
        loss_Q = ((self.Q.T @ self.Q - I_r) ** 2).sum()
        return self.ortho_lambda * (loss_P + loss_Q)

    @torch.no_grad()
    def update_importance(self, beta: float = 0.85) -> None:
        """公式 (3) 重要性打分 EMA 更新。需在 .backward() 之后、optimizer.step() 之前调用。"""
        if self.Lambda.grad is None:
            return
        S = (self.Lambda * self.Lambda.grad).abs().detach()
        self.S_ema.mul_(beta).add_((1 - beta) * S)

    @torch.no_grad()
    def prune_to(self, r_target: int) -> None:
        """保留 S_ema 最大的 r_target 个 λ，其余 mask 置 0。"""
        if r_target >= self.r_init:
            self.mask.fill_(1.0)
            return
        topk = torch.topk(self.S_ema, r_target).indices
        new_mask = torch.zeros_like(self.mask)
        new_mask[topk] = 1.0
        self.mask.copy_(new_mask)

    @property
    def active_rank(self) -> int:
        return int(self.mask.sum().item())


class AdaLoRAGPT2(nn.Module):
    """把 GPT-2 用 AdaLoRALinear 包装。"""

    def __init__(
        self,
        base_model_name: str = "gpt2",
        r_init: int = 12,
        alpha: int = 16,
        ortho_lambda: float = 0.1,
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
            new = AdaLoRALinear(old, r_init=r_init, alpha=alpha, ortho_lambda=ortho_lambda)
            setattr(parent, attr, new)

        self.r_init = r_init
        self.alpha = alpha
        self.ortho_lambda = ortho_lambda

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def total_ortho_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for m in self.lm.modules():
            if isinstance(m, AdaLoRALinear):
                loss = loss + m.ortho_loss()
        return loss

    def update_all_importance(self, beta: float = 0.85) -> None:
        for m in self.lm.modules():
            if isinstance(m, AdaLoRALinear):
                m.update_importance(beta)

    def prune_all_to(self, r_target: int) -> None:
        for m in self.lm.modules():
            if isinstance(m, AdaLoRALinear):
                m.prune_to(r_target)

    def get_all_importance(self) -> torch.Tensor:
        """返回 (n_layers, r_init) 的 S_ema 矩阵，方便可视化。"""
        rows = []
        for m in self.lm.modules():
            if isinstance(m, AdaLoRALinear):
                rows.append(m.S_ema.clone())
        return torch.stack(rows)

    def get_active_ranks(self) -> list[int]:
        """返回每层当前激活的秩。"""
        return [m.active_rank for m in self.lm.modules() if isinstance(m, AdaLoRALinear)]


def cubic_schedule(t: int, t_warmup: int, T: int, r_init: int, r_final: int) -> int:
    """公式 (4): 立方衰减剪枝调度。"""
    if t <= t_warmup:
        return r_init
    if t >= T:
        return r_final
    progress = (t - t_warmup) / (T - t_warmup)
    r = r_final + (r_init - r_final) * (1 - progress) ** 3
    return max(r_final, int(round(r)))


def main() -> None:
    torch.manual_seed(42)
    model = AdaLoRAGPT2(r_init=12, alpha=16)
    print_param_summary(model, "AdaLoRAGPT2 (r_init=12)")
    # Expected trainable: 12 层 × (2304*12 + 12 + 768*12) = 12 × (27648 + 12 + 9216) = 12 × 36876 = 442,512

    print(f"\n参数布局（per layer）：")
    print(f"  P shape: (d_out=2304, r_init=12)  = 27,648")
    print(f"  Λ shape: (r_init=12,)              = 12")
    print(f"  Q shape: (d_in=768, r_init=12)    = 9,216")
    print(f"  subtotal:                         = 36,876")
    print(f"  12 层合计:                        = 442,512")

    print(f"\n剪枝调度演示 (r_init=12, r_final=4, T=1000, t_warmup=100):")
    for t in [0, 100, 200, 500, 800, 1000]:
        r = cubic_schedule(t, 100, 1000, 12, 4)
        print(f"  step={t:>4}: budget_r = {r}")

    enc = model.tokenizer("hello world", return_tensors="pt", padding=True)
    out = model(enc["input_ids"], enc["attention_mask"], labels=enc["input_ids"])
    out.loss.backward()
    model.update_all_importance()
    print(f"\n前向 + 重要性更新一次：")
    print(f"  loss = {out.loss.item():.4f}")
    print(f"  layer 0 S_ema = {model.lm.transformer.h[0].attn.c_attn.S_ema.tolist()}")


if __name__ == "__main__":
    main()
