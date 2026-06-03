"""Switch Transformer router — top-1 极简."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import load_balance_loss


class SwitchRouter(nn.Module):
    def __init__(self, d_model: int, n_experts: int):
        super().__init__()
        self.W = nn.Linear(d_model, n_experts, bias=False)
        self.n_experts = n_experts

    def forward(self, x: torch.Tensor):
        logits = self.W(x)
        gates_all = F.softmax(logits, dim=-1)
        top1_gate, top1_idx = gates_all.max(dim=-1, keepdim=True)
        aux = load_balance_loss(gates_all, top1_idx, self.n_experts)
        return top1_gate, top1_idx, aux


if __name__ == "__main__":
    torch.manual_seed(0)
    r = SwitchRouter(d_model=16, n_experts=4)
    x = torch.randn(20, 16)
    g, idx, aux = r(x)
    print(f"gates {g.shape}  idx {idx.shape}  aux={aux.item():.4f}")
    print(f"expert utilization: {torch.bincount(idx.flatten(), minlength=4).tolist()}")
