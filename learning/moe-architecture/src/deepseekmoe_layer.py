"""DeepSeekMoE Layer — 细粒度 + 共享专家.

教学版：4 routed + 1 shared，top-2（实际 V3 是 256 + 1 / top-8）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_layer_naive import Expert


class DeepSeekMoELayer(nn.Module):
    def __init__(self, d_model: int, n_routed: int = 4, n_shared: int = 1,
                 top_k: int = 2, d_ff_routed: int = 64, d_ff_shared: int = 128):
        super().__init__()
        self.n_routed = n_routed
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_routed, bias=False)
        self.routed = nn.ModuleList(
            Expert(d_model, d_ff_routed) for _ in range(n_routed)
        )
        # shared expert: 始终参与
        self.shared = Expert(d_model, d_ff_shared)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        x_flat = x.view(-1, d)
        # 1. shared 永远过
        out = self.shared(x_flat)
        # 2. routed top-k
        logits = self.gate(x_flat)
        gates_all = F.softmax(logits, dim=-1)
        top_k_gates, top_k_idx = gates_all.topk(self.top_k, dim=-1)
        top_k_gates = top_k_gates / (top_k_gates.sum(-1, keepdim=True) + 1e-9)

        for e_idx in range(self.n_routed):
            mask = (top_k_idx == e_idx)
            if not mask.any():
                continue
            tok_idx, slot = mask.nonzero(as_tuple=True)
            tok_in = x_flat[tok_idx]
            expert_out = self.routed[e_idx](tok_in)
            weight = top_k_gates[tok_idx, slot].unsqueeze(-1)
            out.index_add_(0, tok_idx, expert_out * weight)

        return out.view(b, t, d)


if __name__ == "__main__":
    torch.manual_seed(0)
    layer = DeepSeekMoELayer(d_model=32, n_routed=4, top_k=2)
    x = torch.randn(2, 16, 32)
    y = layer(x)
    print(f"out {y.shape}, params {sum(p.numel() for p in layer.parameters())}")
