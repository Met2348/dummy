"""MoE Layer naive — Shazeer 2017 style + top-2 routing + aux loss.

教学目标：
    1. 完整 MoE forward 流程
    2. gating + top-k
    3. aux load balance loss
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import load_balance_loss


class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))


class MoELayer(nn.Module):
    """Top-k routing MoE."""
    def __init__(self, d_model: int, n_experts: int, top_k: int = 2,
                 d_ff: int | None = None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(
            Expert(d_model, d_ff) for _ in range(n_experts)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (b, t, d). Return (out, aux_loss)."""
        b, t, d = x.shape
        x_flat = x.view(-1, d)                              # (n_tok, d)
        logits = self.gate(x_flat)                           # (n_tok, n_experts)
        gates_all = F.softmax(logits, dim=-1)
        top_k_gates, top_k_idx = gates_all.topk(self.top_k, dim=-1)
        # 归一化 top_k gates → sum=1
        top_k_gates = top_k_gates / (top_k_gates.sum(-1, keepdim=True) + 1e-9)

        out_flat = torch.zeros_like(x_flat)
        for e_idx in range(self.n_experts):
            # 找出选了 e_idx 的 (token, slot) 对
            mask = (top_k_idx == e_idx)              # (n_tok, top_k)
            if not mask.any():
                continue
            tok_idx, slot = mask.nonzero(as_tuple=True)
            tok_input = x_flat[tok_idx]
            expert_out = self.experts[e_idx](tok_input)
            weight = top_k_gates[tok_idx, slot].unsqueeze(-1)
            out_flat.index_add_(0, tok_idx, expert_out * weight)

        aux = load_balance_loss(gates_all, top_k_idx, self.n_experts)
        return out_flat.view(b, t, d), aux


if __name__ == "__main__":
    torch.manual_seed(0)
    layer = MoELayer(d_model=32, n_experts=4, top_k=2, d_ff=64)
    x = torch.randn(2, 16, 32)
    y, aux = layer(x)
    print(f"out {y.shape}, aux {aux.item():.4f}")
    print(f"params {sum(p.numel() for p in layer.parameters())}")
