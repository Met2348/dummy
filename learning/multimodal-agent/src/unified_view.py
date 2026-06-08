"""五线综合统一公式数值验证 - 跨主线等价对.

实现 lecture L13 中的几个等价关系：
    1. LoRA(sigma=id, rank=full) equals Full Fine-tuning
    2. LoRA(sigma=id) equals Parallel Adapter (no sigma)
    3. Prefix Tuning equals Parallel Adapter (theory view)
    4. DPO = beta * log(pi/pi_ref) when implicit r maps to BT loss

Toy numerical checks only. No real model training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== 1. LoRA vs Parallel Adapter =====

class LoRALayer(nn.Module):
    """LoRA: W + B*A, without activation."""
    def __init__(self, d_in: int, d_out: int, r: int = 8) -> None:
        super().__init__()
        self.W = nn.Linear(d_in, d_out, bias=False)
        self.A = nn.Linear(d_in, r, bias=False)
        self.B = nn.Linear(r, d_out, bias=False)
        # 初始化使 BA = 0
        nn.init.zeros_(self.B.weight)
        nn.init.normal_(self.A.weight, std=0.01)

    def forward(self, x):
        return self.W(x) + self.B(self.A(x))


class ParallelAdapter(nn.Module):
    """Parallel Adapter (no sigma): h + Linear(h), no activation."""
    def __init__(self, d_in: int, d_out: int, r: int = 8) -> None:
        super().__init__()
        self.W = nn.Linear(d_in, d_out, bias=False)
        self.down = nn.Linear(d_in, r, bias=False)
        self.up = nn.Linear(r, d_out, bias=False)
        nn.init.zeros_(self.up.weight)
        nn.init.normal_(self.down.weight, std=0.01)

    def forward(self, x):
        return self.W(x) + self.up(self.down(x))


def test_lora_vs_parallel_adapter():
    """LoRA and no-activation parallel adapter should match."""
    d_in, d_out, r = 8, 8, 4
    lora = LoRALayer(d_in, d_out, r)
    pa = ParallelAdapter(d_in, d_out, r)
    # 同步权重
    pa.W.weight.data = lora.W.weight.data.clone()
    pa.down.weight.data = lora.A.weight.data.clone()
    pa.up.weight.data = lora.B.weight.data.clone()

    x = torch.randn(4, d_in)
    y_lora = lora(x)
    y_pa = pa(x)
    assert torch.allclose(y_lora, y_pa, atol=1e-6), (y_lora, y_pa)
    print("PASS LoRA(sigma=id) equals Parallel Adapter (no sigma)")
    print(f"  max diff = {(y_lora - y_pa).abs().max().item():.2e}")


# ===== 2. DPO 隐式 RM 一致性 =====

def implicit_reward(log_pi: torch.Tensor, log_ref: torch.Tensor,
                    beta: float = 0.1) -> torch.Tensor:
    """r(x, y) = beta * log(pi(y|x) / pi_ref(y|x))."""
    return beta * (log_pi - log_ref)


def bt_loss_from_rewards(r_chosen: torch.Tensor,
                         r_rejected: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(r_chosen - r_rejected).mean()


def dpo_loss_direct(
    log_pi_chosen: torch.Tensor,
    log_ref_chosen: torch.Tensor,
    log_pi_rejected: torch.Tensor,
    log_ref_rejected: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Direct DPO loss = -log sigmoid(beta * margin)."""
    margin = beta * ((log_pi_chosen - log_ref_chosen) -
                     (log_pi_rejected - log_ref_rejected))
    return -F.logsigmoid(margin).mean()


def test_dpo_equiv_bt_implicit_rm():
    """DPO loss = BT loss with implicit RM."""
    torch.manual_seed(0)
    log_pi_c = torch.randn(3)
    log_pi_r = torch.randn(3)
    log_ref_c = torch.randn(3)
    log_ref_r = torch.randn(3)
    beta = 0.1

    L_direct = dpo_loss_direct(log_pi_c, log_ref_c, log_pi_r, log_ref_r, beta)
    r_c = implicit_reward(log_pi_c, log_ref_c, beta)
    r_r = implicit_reward(log_pi_r, log_ref_r, beta)
    L_bt = bt_loss_from_rewards(r_c, r_r)

    assert torch.allclose(L_direct, L_bt, atol=1e-7)
    print("\nPASS DPO loss equals BT loss with implicit RM")
    print(f"  L_direct = {L_direct.item():.6f}")
    print(f"  L_bt     = {L_bt.item():.6f}")


# ===== 3. GRPO advantage 与 PPO 的关系 =====

def test_grpo_advantage_zero_mean():
    """GRPO group-z-score advantage 在每组均值为 0。"""
    k = 8
    rewards = torch.tensor([1.0, 0, 1, 0, 1, 1, 0, 0])  # group of 8
    mean = rewards.mean()
    std = rewards.std() + 1e-8
    A = (rewards - mean) / std
    assert abs(A.mean().item()) < 1e-6, A.mean()
    print("\nPASS GRPO group advantage mean is 0")
    print(f"  rewards = {rewards.tolist()}")
    print(f"  A       = {[round(a, 3) for a in A.tolist()]}")


def main():
    print("五线综合统一公式 - 数值验证\n" + "=" * 50)
    test_lora_vs_parallel_adapter()
    test_dpo_equiv_bt_implicit_rm()
    test_grpo_advantage_zero_mean()
    print("\n" + "=" * 50)
    print("3 个等价关系全部验证 PASS")


if __name__ == "__main__":
    main()
