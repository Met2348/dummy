"""PiSSA 最小实现（手写）。

对应论文：Meng et al. 2024, arXiv:2404.02948
对应 lecture: lectures/03-pissa.md

核心：
  1. 对 W_0 做 SVD: W_0 = U Σ V^T
  2. 用 top-r 主成分初始化: B = U[:r] sqrt(Σ[:r]), A = sqrt(Σ[:r]) V^T[:r]
  3. W_res = W_0 - BA 作为冻结基础
  4. 训练 forward: h = W_res(x) + α/r BA(x)

与 LoRA 的差异:
  - LoRA: base = W_0, B 零初始化
  - PiSSA: base = W_res, B、A 用 SVD 主成分初始化

公式索引:
    W_0 = U Σ V^T                                (1)
    B = U_:r √Σ_:r,  A = √Σ_:r V^T_:r            (2)
    W_res = W_0 - BA                             (3)
    h = W_res(x) + α/r BA(x)                     (4)
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


def _extract_weight(base_linear: nn.Module) -> torch.Tensor:
    """统一返回 (out, in) 形状的权重张量。"""
    if is_conv1d(base_linear):
        return base_linear.weight.data.T.clone()  # Conv1D.weight is (in, out)
    return base_linear.weight.data.clone()  # nn.Linear.weight is (out, in)


def _write_back(base_linear: nn.Module, W_out_in: torch.Tensor) -> None:
    """把 (out, in) 张量写回 base_linear.weight，处理 Conv1D 转置。"""
    if is_conv1d(base_linear):
        base_linear.weight.data.copy_(W_out_in.T)
    else:
        base_linear.weight.data.copy_(W_out_in)


class PiSSALinear(nn.Module):
    """单层 PiSSA。"""

    def __init__(
        self,
        base_linear: nn.Module,
        r: int = 8,
        alpha: int | None = None,
        init_method: str = "pissa",  # 或 "olora"
    ):
        super().__init__()
        if alpha is None:
            alpha = r  # PiSSA 论文默认 α = r → scaling = 1

        d_in, d_out = get_in_out_dims(base_linear)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.is_conv1d = is_conv1d(base_linear)

        W = _extract_weight(base_linear).float()  # (d_out, d_in)
        if init_method == "pissa":
            # 公式 (1): SVD
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            # 公式 (2): top-r 主成分平均分配 sqrt
            sqrt_S = S[:r].sqrt()
            B_init = U[:, :r] * sqrt_S.unsqueeze(0)  # (d_out, r)
            A_init = sqrt_S.unsqueeze(-1) * Vt[:r, :]  # (r, d_in)
        elif init_method == "olora":
            # OLoRA: QR 分解
            # torch.linalg.qr 返回 Q (d_out, d_out 或 d_in), R (d_in, d_in)
            Q, R = torch.linalg.qr(W, mode="reduced")
            B_init = Q[:, :r]  # (d_out, r)
            A_init = R[:r, :]  # (r, d_in)
        else:
            raise ValueError(f"Unknown init_method: {init_method}")

        # 公式 (3): W_res = W_0 - BA
        W_res = W - B_init @ A_init

        # 写回 base.weight
        _write_back(base_linear, W_res.to(base_linear.weight.dtype))
        for p in base_linear.parameters():
            p.requires_grad = False
        self.base = base_linear

        # 可训练的 A、B
        self.A = nn.Parameter(A_init.clone())
        self.B = nn.Parameter(B_init.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """公式 (4) 前向。"""
        base_out = self.base(x)
        # LoRA 风格: x @ A.T @ B.T
        lora_out = x @ self.A.T @ self.B.T
        return base_out + self.scaling * lora_out

    @torch.no_grad()
    def reconstruct_W0(self) -> torch.Tensor:
        """返回 W_res + BA = W_0（用于一致性测试）。"""
        W_res = _extract_weight(self.base)
        return W_res + self.scaling * (self.B @ self.A)


class PiSSAGPT2(nn.Module):
    """GPT-2 + PiSSA（或 OLoRA）。"""

    def __init__(
        self,
        base_model_name: str = "gpt2",
        r: int = 8,
        alpha: int | None = None,
        target_modules: tuple[str, ...] = ("c_attn",),
        init_method: str = "pissa",
    ):
        super().__init__()
        self.lm = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        freeze_base_model(self.lm)

        matches = target_linear_modules(self.lm, target_modules)
        for qname, old in matches:
            parent, attr = get_parent_and_attr(self.lm, qname)
            new = PiSSALinear(old, r=r, alpha=alpha, init_method=init_method)
            setattr(parent, attr, new)

        self.r = r
        self.alpha = alpha if alpha is not None else r
        self.init_method = init_method

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )


def main() -> None:
    torch.manual_seed(42)
    model = PiSSAGPT2(r=8)
    print_param_summary(model, "PiSSAGPT2 (r=8, init=pissa)")

    # 验证：W_res + BA == W_0（初始重建）
    layer0 = model.lm.transformer.h[0].attn.c_attn
    base_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
    W_0 = base_gpt2.transformer.h[0].attn.c_attn.weight.data.T  # (out, in)
    W_reconstructed = layer0.reconstruct_W0()
    diff = (W_0 - W_reconstructed).abs().max().item()
    print(f"\n初始重建测试: |W_res + BA - W_0|.max = {diff:.4e}")
    assert diff < 1e-3, f"重建误差过大: {diff}"
    print(f"  [OK] 初始 W_res + BA = W_0（与原 GPT-2 等价）")

    # 比较 forward 与原始 GPT-2
    enc = model.tokenizer("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out_pissa = model(enc["input_ids"], enc["attention_mask"])
        out_base = base_gpt2(enc["input_ids"], attention_mask=enc["attention_mask"])
    fwd_diff = (out_pissa.logits - out_base.logits).abs().max().item()
    print(f"\nforward 一致性: |logits_pissa - logits_base|.max = {fwd_diff:.4e}")
    assert fwd_diff < 1e-1, f"forward 差异过大: {fwd_diff}"
    print(f"  [OK] PiSSA 训练开始时 forward ≈ 原始 GPT-2")


if __name__ == "__main__":
    main()
