"""Aux-Loss-Free routing — DeepSeek-V3 2024.12 创新.

教学目标：
    1. 偏置项 bias 控制 expert 负载（不进梯度）
    2. update_rate=1e-3，bias_init=0 严格按论文
    3. 无 aux loss，与 ce_loss 解耦
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxFreeRouter(nn.Module):
    def __init__(self, d_model: int, n_experts: int, top_k: int = 2,
                 update_rate: float = 1e-3):
        super().__init__()
        self.W = nn.Linear(d_model, n_experts, bias=False)
        # bias 是 buffer (不进 gradient)
        self.register_buffer("bias", torch.zeros(n_experts))
        self.n_experts = n_experts
        self.top_k = top_k
        self.update_rate = update_rate

    def forward(self, x: torch.Tensor):
        """x: (n_tok, d). 返回 (top_k_gates, top_k_idx, None)."""
        logits = self.W(x)
        scores = F.softmax(logits, dim=-1)
        # bias 影响排序，但不影响 gates
        sort_scores = scores + self.bias
        _, top_k_idx = sort_scores.topk(self.top_k, dim=-1)
        top_k_gates = scores.gather(-1, top_k_idx)
        top_k_gates = top_k_gates / (top_k_gates.sum(-1, keepdim=True) + 1e-9)

        if self.training:
            self._update_bias(top_k_idx, x.shape[0])
        return top_k_gates, top_k_idx, None    # NO aux loss

    @torch.no_grad()
    def _update_bias(self, top_k_idx: torch.Tensor, n_tok: int) -> None:
        load = torch.bincount(top_k_idx.flatten(),
                              minlength=self.n_experts).float()
        target = self.top_k * n_tok / self.n_experts
        # 高于 target → bias 降; 低于 → bias 升
        delta = torch.zeros_like(self.bias)
        delta[load > target] = -self.update_rate
        delta[load < target] = +self.update_rate
        self.bias.add_(delta)


def demo_balance() -> None:
    """100 step 后 expert load 应平衡."""
    torch.manual_seed(0)
    router = AuxFreeRouter(d_model=16, n_experts=4, top_k=2)
    opt = torch.optim.Adam(router.W.parameters(), lr=1e-3)
    router.train()
    for step in range(100):
        x = torch.randn(64, 16)
        gates, idx, _ = router(x)
        # 玩具 loss (不重要)
        loss = (gates ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step in {0, 10, 50, 99}:
            load = torch.bincount(idx.flatten(), minlength=4).tolist()
            print(f"step {step:3d}  load {load}  bias {router.bias.tolist()}")


if __name__ == "__main__":
    demo_balance()
