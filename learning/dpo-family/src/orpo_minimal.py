"""ORPO (Odds Ratio Preference Optimization, Hong 2024) — 无 reference model.

关键观察：odds 比 (p/(1-p)) 可以替代 log π/π_ref 充当 reward.
loss = L_SFT + λ · L_OR
    L_OR = -log sigmoid(log odds_chosen - log odds_rejected)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def log_odds(log_p: torch.Tensor) -> torch.Tensor:
    """log(p/(1-p)) = log p - log(1-p).

    数值稳定：log(1-p) = log(1 - exp(log p)) using log1p(-exp(...))
    """
    # 截断防溢出
    log_p = log_p.clamp(max=-1e-6)
    return log_p - torch.log1p(-log_p.exp())


def orpo_loss(
    log_p_chosen: torch.Tensor,        # SFT NLL of chosen
    log_p_rejected: torch.Tensor,      # SFT NLL of rejected
    sft_loss: torch.Tensor,            # standard SFT loss
    lambda_or: float = 0.1,
) -> torch.Tensor:
    """ORPO = SFT loss + λ · log_odds_ratio penalty."""
    or_c = log_odds(log_p_chosen)
    or_r = log_odds(log_p_rejected)
    L_or = -F.logsigmoid(or_c - or_r).mean()
    return sft_loss + lambda_or * L_or


if __name__ == "__main__":
    print("ORPO minimal — smoke test")
    torch.manual_seed(0)
    B = 4
    log_p_c = -torch.rand(B) * 5      # negative log prob
    log_p_r = -torch.rand(B) * 5 - 1   # rejected 更负
    sft_l = torch.tensor(2.5)
    L = orpo_loss(log_p_c, log_p_r, sft_l, lambda_or=0.1)
    print(f"  ORPO loss = {L.item():.4f}")
    print("  关键：无 ref model — 显存省一半")
