"""Expert Choice routing — Zhou 2022 反向路由."""
from __future__ import annotations

import torch
import torch.nn as nn


class ExpertChoiceRouter(nn.Module):
    def __init__(self, d_model: int, n_experts: int, capacity_per_expert: int):
        super().__init__()
        self.W = nn.Linear(d_model, n_experts, bias=False)
        self.n_experts = n_experts
        self.cap = capacity_per_expert

    def forward(self, x: torch.Tensor):
        """x: (n_tok, d). 返回 (expert_gates, token_idx, token_util)."""
        logits = self.W(x).T                          # (n_expert, n_tok)
        cap = min(self.cap, logits.shape[-1])
        gates, token_idx = logits.topk(cap, dim=-1)    # (n_expert, cap)
        gates = gates.softmax(dim=-1)
        # token utilization：被选中的 token 比例
        chosen = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        chosen[token_idx.flatten()] = True
        token_util = chosen.float().mean().item()
        return gates, token_idx, token_util


if __name__ == "__main__":
    torch.manual_seed(0)
    r = ExpertChoiceRouter(d_model=16, n_experts=4, capacity_per_expert=6)
    x = torch.randn(20, 16)
    g, idx, util = r(x)
    print(f"expert_gates {g.shape}  token_idx {idx.shape}")
    print(f"token utilization: {util:.2%}")
