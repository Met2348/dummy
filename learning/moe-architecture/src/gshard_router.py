"""GShard router — top-2 + capacity factor + aux loss."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import load_balance_loss, capacity


class GShardRouter(nn.Module):
    def __init__(self, d_model: int, n_experts: int, top_k: int = 2,
                 capacity_factor: float = 1.25):
        super().__init__()
        self.W = nn.Linear(d_model, n_experts, bias=False)
        self.n_experts = n_experts
        self.top_k = top_k
        self.cap_f = capacity_factor

    def forward(self, x: torch.Tensor):
        """x: (n_tok, d). Returns (gates_kept, idx_kept, aux_loss, drop_mask)."""
        n_tok, d = x.shape
        logits = self.W(x)
        gates_all = F.softmax(logits, dim=-1)
        top_k_gates, top_k_idx = gates_all.topk(self.top_k, dim=-1)
        # capacity drop（按选中顺序，超出 capacity 的 slot 标记 drop）
        cap = capacity(n_tok, self.n_experts, self.top_k, factor=self.cap_f)
        # 简化版: 对每 expert 取前 cap 个 (按概率排序)
        drop_mask = torch.zeros_like(top_k_idx, dtype=torch.bool)
        for e in range(self.n_experts):
            choosers = (top_k_idx == e)               # (n_tok, top_k)
            if choosers.sum() <= cap:
                continue
            # 按 gate 排序，超出的 drop
            tok_idx, slot = choosers.nonzero(as_tuple=True)
            probs = top_k_gates[tok_idx, slot]
            order = probs.argsort(descending=True)
            to_drop = order[cap:]
            drop_mask[tok_idx[to_drop], slot[to_drop]] = True
        # aux loss
        aux = load_balance_loss(gates_all, top_k_idx, self.n_experts)
        return top_k_gates, top_k_idx, aux, drop_mask


if __name__ == "__main__":
    torch.manual_seed(0)
    router = GShardRouter(d_model=16, n_experts=4, top_k=2,
                          capacity_factor=1.25)
    x = torch.randn(20, 16)
    g, idx, aux, drop = router(x)
    print(f"gates {g.shape}  idx {idx.shape}  aux={aux.item():.4f}  drop={drop.sum().item()}")
