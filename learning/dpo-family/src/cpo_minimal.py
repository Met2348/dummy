"""CPO (Contrastive PO, Xu 2024) — Contrastive + SFT.

loss = L_SFT(chosen) + λ · L_DPO_no_ref
其中 L_DPO_no_ref 用 actor 自身代替 ref（省 ref）.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def cpo_loss(
    log_p_chosen_actor: torch.Tensor,    # sum log π_chosen
    log_p_rejected_actor: torch.Tensor,  # sum log π_rejected
    sft_loss_chosen: torch.Tensor,       # standard NLL on chosen
    beta: float = 0.1,
    lambda_c: float = 0.5,
) -> torch.Tensor:
    """CPO = NLL(chosen) + λ · DPO-style margin (no ref)."""
    margin = beta * (log_p_chosen_actor - log_p_rejected_actor)
    L_contrast = -F.logsigmoid(margin).mean()
    return sft_loss_chosen + lambda_c * L_contrast


if __name__ == "__main__":
    print("CPO minimal — smoke test")
    torch.manual_seed(0)
    B = 4
    log_p_c = -torch.rand(B) * 3
    log_p_r = -torch.rand(B) * 3 - 1
    sft_l = torch.tensor(2.0)
    L = cpo_loss(log_p_c, log_p_r, sft_l, beta=0.1, lambda_c=0.5)
    print(f"  CPO loss = {L.item():.4f}")
    print("  关键：SFT + 对比，无 ref")
