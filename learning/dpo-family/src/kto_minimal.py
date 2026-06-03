"""KTO (Kahneman-Tversky Optimization, Ethayarajh 2024) — 单边偏好.

不需要 chosen/rejected pair，只需要 (y, is_desirable) 标签.
基于前景理论：损失敏感度 > 收益敏感度。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kto_loss(
    log_p_actor: torch.Tensor,        # (B,) 每个样本 sum log π over response
    log_p_ref: torch.Tensor,           # (B,)
    is_desirable: torch.Tensor,        # (B,) 1=desired, 0=undesired
    beta: float = 0.1,
    lambda_d: float = 1.0,             # desired 权重
    lambda_u: float = 1.0,             # undesired 权重
) -> torch.Tensor:
    """KTO loss.

    desired:    L = λ_d · (1 - sigmoid(β·(r - z_ref)))
    undesired:  L = λ_u · (1 - sigmoid(β·(z_ref - r)))
    z_ref = mean KL(actor || ref) (近似 0 起步).
    """
    log_ratio = log_p_actor - log_p_ref
    z_ref = log_ratio.detach().mean().clamp(min=0)  # 近似

    is_desirable = is_desirable.float()
    val_d = lambda_d * (1 - torch.sigmoid(beta * (log_ratio - z_ref)))
    val_u = lambda_u * (1 - torch.sigmoid(beta * (z_ref - log_ratio)))
    loss = is_desirable * val_d + (1 - is_desirable) * val_u
    return loss.mean()


if __name__ == "__main__":
    print("KTO minimal — smoke test")
    torch.manual_seed(0)
    B = 8
    log_p_actor = torch.randn(B)
    log_p_ref = torch.randn(B)
    is_desired = torch.randint(0, 2, (B,))
    L = kto_loss(log_p_actor, log_p_ref, is_desired)
    print(f"  desired count = {is_desired.sum().item()}/{B}")
    print(f"  KTO loss = {L.item():.4f}")
    print("  特点：不需要 pair，标 desired/undesired 即可")
