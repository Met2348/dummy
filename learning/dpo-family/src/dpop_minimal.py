"""DPOP (Pal 2024) — 修 DPO 的 chosen 概率下降问题.

DPO 训练时，chosen 概率有时也会下降（因为 loss 只追 margin）.
DPOP 加 hinge 项：max(0, log_ratio_ref_chosen - log_ratio_actor_chosen).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dpop_loss(
    log_p_c_actor: torch.Tensor, log_p_c_ref: torch.Tensor,
    log_p_r_actor: torch.Tensor, log_p_r_ref: torch.Tensor,
    beta: float = 0.1,
    lambda_p: float = 50.0,  # 高权重 hinge
) -> torch.Tensor:
    """DPOP = DPO + λ·max(0, log_p_c_ref - log_p_c_actor)."""
    log_ratio_c = log_p_c_actor - log_p_c_ref
    log_ratio_r = log_p_r_actor - log_p_r_ref
    margin = beta * (log_ratio_c - log_ratio_r)
    L_dpo = -F.logsigmoid(margin).mean()
    # hinge：要求 actor 给 chosen 的概率 ≥ ref 给 chosen 的概率
    hinge = F.relu(log_p_c_ref - log_p_c_actor).mean()
    return L_dpo + lambda_p * hinge


def demo_chosen_prob_drop():
    """演示 DPO chosen 概率下降反例."""
    print("演示 DPO chosen prob 下降:")
    # 反例：actor 把 chosen 概率从 0.5 降到 0.3，但 rejected 从 0.4 降到 0.1
    log_p_c_actor = torch.log(torch.tensor([0.3]))
    log_p_c_ref = torch.log(torch.tensor([0.5]))
    log_p_r_actor = torch.log(torch.tensor([0.1]))
    log_p_r_ref = torch.log(torch.tensor([0.4]))
    L_dpo_alone = -F.logsigmoid(0.1 * ((log_p_c_actor - log_p_c_ref) - (log_p_r_actor - log_p_r_ref))).mean()
    L_dpop = dpop_loss(log_p_c_actor, log_p_c_ref, log_p_r_actor, log_p_r_ref)
    print(f"  DPO loss alone = {L_dpo_alone.item():.4f} (chosen 降也能赢)")
    print(f"  DPOP loss = {L_dpop.item():.4f} (hinge 惩罚 chosen 降)")


if __name__ == "__main__":
    print("DPOP minimal\n" + "=" * 50)
    demo_chosen_prob_drop()
