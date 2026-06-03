"""Safe-RLHF (PKU 2023) — Lagrangian 多目标优化.

idea: helpful 与 harmless 双 reward, Lagrangian 拉格朗日松弛求平衡.

L = E[R_helpful] - λ · max(0, c_harmless - R_harmless)
其中 c_harmless 是 harmless 约束 (e.g. 期望 0.8 以上).
λ 自动调节: 若违反约束 λ↑, 满足则 λ↓.
"""
from __future__ import annotations

import torch


class LagrangianSafeRLHF:
    """Lagrangian dual ascent."""

    def __init__(self, harmless_threshold: float = 0.8, lr_lambda: float = 0.01):
        self.c = harmless_threshold
        self.lr_lambda = lr_lambda
        self.lam = torch.tensor(0.1, requires_grad=False)

    def policy_loss(self, r_helpful: torch.Tensor, r_harmless: torch.Tensor) -> torch.Tensor:
        violation = (self.c - r_harmless).clamp(min=0)
        return -(r_helpful - self.lam * violation).mean()

    def update_lambda(self, r_harmless: torch.Tensor):
        """λ 上升若 harmless 不足，下降若超出."""
        gap = self.c - r_harmless.mean().item()
        self.lam = (self.lam + self.lr_lambda * gap).clamp(min=0)


def maxmin_rlhf(r_helpful: torch.Tensor, r_harmless: torch.Tensor) -> torch.Tensor:
    """简单 maxmin: 最大化两者最小值."""
    return torch.minimum(r_helpful, r_harmless)


if __name__ == "__main__":
    print("Safe-RLHF Lagrangian demo\n" + "=" * 50)
    safe = LagrangianSafeRLHF(harmless_threshold=0.8)

    # 模拟 100 step
    torch.manual_seed(0)
    history = []
    for step in range(50):
        # 模拟 actor 偶尔产生 harm
        r_h = torch.rand(8) * 0.8 + 0.1
        r_safe = torch.rand(8) * 0.6 + 0.2 + step * 0.005
        L = safe.policy_loss(r_h, r_safe)
        history.append({
            "step": step, "loss": L.item(),
            "r_helpful": r_h.mean().item(),
            "r_harmless": r_safe.mean().item(),
            "lambda": safe.lam.item(),
        })
        safe.update_lambda(r_safe)
        if step % 10 == 0:
            print(f"  step {step:3d}: L={L.item():+.3f} R_h={r_h.mean():.2f} "
                  f"R_safe={r_safe.mean():.2f} λ={safe.lam.item():.3f}")
    print("\n趋势: harmless 上升 → λ 下降; 不达标时 λ 上升强制收敛.")

    print("\n[MaxMin-RLHF 对比]")
    r_max = maxmin_rlhf(torch.tensor([0.9, 0.3]), torch.tensor([0.4, 0.8]))
    print(f"  helpful=[0.9, 0.3], harmless=[0.4, 0.8] → maxmin={r_max.tolist()}")
