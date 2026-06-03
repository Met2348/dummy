"""SimPO (Meng 2024) — length-normalized + 无 ref model.

reward = (1/|y|) · log π(y|x) — γ
(γ 是 target margin，超参)

loss = -log sigmoid(β·(r_chosen - r_rejected))
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def length_normalized_logp(log_p_per_token: torch.Tensor,
                           response_mask: torch.Tensor) -> torch.Tensor:
    """(1/|y|) · sum log π."""
    seq_logp = (log_p_per_token * response_mask).sum(dim=1)
    seq_len = response_mask.sum(dim=1).clamp(min=1)
    return seq_logp / seq_len


def simpo_loss(
    log_p_chosen: torch.Tensor,        # (B, T_c) per token
    mask_chosen: torch.Tensor,         # (B, T_c)
    log_p_rejected: torch.Tensor,      # (B, T_r) per token
    mask_rejected: torch.Tensor,       # (B, T_r)
    beta: float = 2.5,
    gamma: float = 1.0,
) -> torch.Tensor:
    r_c = length_normalized_logp(log_p_chosen, mask_chosen)
    r_r = length_normalized_logp(log_p_rejected, mask_rejected)
    margin = beta * (r_c - r_r) - gamma
    return -F.logsigmoid(margin).mean()


if __name__ == "__main__":
    print("SimPO minimal — smoke test")
    torch.manual_seed(0)
    B, T = 2, 10
    log_p_c = torch.randn(B, T) - 1     # log π < 0
    log_p_r = torch.randn(B, T) - 1.5
    mask_c = torch.tensor([[1] * 8 + [0] * 2,
                           [1] * 6 + [0] * 4], dtype=torch.float32)
    mask_r = torch.tensor([[1] * 10,
                           [1] * 9 + [0]], dtype=torch.float32)
    L = simpo_loss(log_p_c, mask_c, log_p_r, mask_r, beta=2.5, gamma=1.0)
    print(f"  SimPO loss = {L.item():.4f}")
    print("  关键：length norm 防止 reward 偏向长 response")
    print("  关键：无 ref model — 显存省一半 + 比 ORPO 简单")
