"""IPO (Identity-PO, Azar 2023) — 修 DPO 偏向极端样本.

DPO 在 chosen/rejected margin 太大时仍然推优化方向，导致 over-fit.
IPO loss = (h - 1/(2β))^2，其中 h = logπ_c/π_r_c - logπ_r/π_r_r
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def ipo_loss(
    log_p_c_actor: torch.Tensor, log_p_c_ref: torch.Tensor,
    log_p_r_actor: torch.Tensor, log_p_r_ref: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """IPO: squared loss 防止 over-confidence."""
    log_ratio_c = log_p_c_actor - log_p_c_ref
    log_ratio_r = log_p_r_actor - log_p_r_ref
    h = log_ratio_c - log_ratio_r
    target = 1.0 / (2 * beta)
    return ((h - target) ** 2).mean()


def dpo_loss(
    log_p_c_actor: torch.Tensor, log_p_c_ref: torch.Tensor,
    log_p_r_actor: torch.Tensor, log_p_r_ref: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    margin = beta * ((log_p_c_actor - log_p_c_ref) - (log_p_r_actor - log_p_r_ref))
    return -F.logsigmoid(margin).mean()


def compare_dpo_ipo_at_high_margin():
    """高 margin 时 DPO 持续推，IPO 饱和."""
    print("DPO vs IPO 在高 margin 行为对比")
    beta = 0.1
    for log_ratio_c, log_ratio_r in [(0.5, -0.5), (2.0, -2.0), (10.0, -10.0)]:
        c = torch.tensor([log_ratio_c]); r = torch.tensor([log_ratio_r])
        zero = torch.zeros_like(c)
        L_dpo = dpo_loss(c, zero, r, zero, beta)
        L_ipo = ipo_loss(c, zero, r, zero, beta)
        print(f"  margin {log_ratio_c - log_ratio_r:+.1f}: DPO={L_dpo.item():.4f} IPO={L_ipo.item():.4f}")


if __name__ == "__main__":
    print("IPO minimal\n" + "=" * 50)
    compare_dpo_ipo_at_high_margin()
