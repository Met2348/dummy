"""LoftQ 最小实现（手写）。

对应论文：Li et al. 2023, arXiv:2310.08659
对应 lecture: lectures/07-loftq.md

核心：交替最小化 ||W - Q - BA||_F²
  init: Q = NF4(W), BA = 0
  loop T 次:
    Step 1: BA = SVD_r(W - Q)
    Step 2: Q = NF4(W - BA)
"""
from __future__ import annotations

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


def _extract_weight(base_linear: nn.Module) -> torch.Tensor:
    if is_conv1d(base_linear):
        return base_linear.weight.data.T.clone()
    return base_linear.weight.data.clone()


def _write_back(base_linear: nn.Module, W: torch.Tensor) -> None:
    if is_conv1d(base_linear):
        base_linear.weight.data.copy_(W.T)
    else:
        base_linear.weight.data.copy_(W)


class LoftQLinear(nn.Module):
    """单层 LoftQ。"""

    def __init__(
        self,
        base_linear: nn.Module,
        r: int = 8,
        alpha: int | None = None,
        n_iter: int = 5,
        block_size: int = 64,
    ):
        super().__init__()
        if alpha is None:
            alpha = r  # 默认 α = r（与 PiSSA 一致）
        d_in, d_out = get_in_out_dims(base_linear)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.is_conv1d = is_conv1d(base_linear)
        self.n_iter = n_iter

        W = _extract_weight(base_linear).float()  # (d_out, d_in)

        # 初始化: Q = NF4(W), BA = 0
        Q = nf4_quant_dequant(W, block_size=block_size)
        BA = torch.zeros_like(W)
        history = [(W - Q - BA).norm().item()]

        B_t = torch.zeros(d_out, r, dtype=W.dtype, device=W.device)
        A_t = torch.zeros(r, d_in, dtype=W.dtype, device=W.device)

        # 交替最小化
        for _ in range(n_iter):
            # Step 1: SVD of (W - Q)
            U, S, Vt = torch.linalg.svd(W - Q, full_matrices=False)
            sqrt_S = S[:r].sqrt()
            B_t = U[:, :r] * sqrt_S.unsqueeze(0)
            A_t = sqrt_S.unsqueeze(-1) * Vt[:r, :]
            BA = B_t @ A_t
            # Step 2: NF4 quantize (W - BA)
            Q = nf4_quant_dequant(W - BA, block_size=block_size)
            history.append((W - Q - BA).norm().item())

        # 残差 base.weight = Q（冻结）
        _write_back(base_linear, Q.to(base_linear.weight.dtype))
        for p in base_linear.parameters():
            p.requires_grad = False
        self.base = base_linear

        # LoRA: 用最后的 (B_t, A_t) 作初始化
        self.A = nn.Parameter(A_t.clone())
        self.B = nn.Parameter(B_t.clone())

        # 保存收敛历史供 debug
        self.register_buffer("convergence_history", torch.tensor(history))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = x @ self.A.T @ self.B.T
        return base_out + self.scaling * lora_out

    def get_convergence(self) -> list[float]:
        return self.convergence_history.tolist()


class LoftQGPT2(nn.Module):
    """GPT-2 + LoftQ。"""

    def __init__(
        self,
        base_model_name: str = "gpt2",
        r: int = 8,
        alpha: int | None = None,
        n_iter: int = 5,
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
            new = LoftQLinear(old, r=r, alpha=alpha, n_iter=n_iter, block_size=block_size)
            setattr(parent, attr, new)

        self.r = r
        self.n_iter = n_iter

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def get_layer_convergence(self, layer_idx: int = 0) -> list[float]:
        return self.lm.transformer.h[layer_idx].attn.c_attn.get_convergence()


def main() -> None:
    print("LoftQ 演示：n_iter=5 交替最小化")
    torch.manual_seed(42)
    model = LoftQGPT2(r=8, alpha=8, n_iter=5)
    print_param_summary(model, "LoftQ (r=8, T=5)")

    # 取 layer 0 看收敛历史
    history = model.get_layer_convergence(0)
    print(f"\nlayer 0 收敛历史 ||W - Q - BA||_F:")
    for t, v in enumerate(history):
        marker = " ← init" if t == 0 else f" (after iter {t})"
        print(f"  t={t}: {v:.4f}{marker}")
    print(f"  最终 / 初始: {history[-1] / history[0]:.4f}")

    enc = model.tokenizer("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"])
    print(f"\nforward logits.shape={tuple(out.logits.shape)}")


if __name__ == "__main__":
    main()
