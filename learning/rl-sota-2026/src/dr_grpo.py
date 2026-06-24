"""Dr. GRPO (Sea AI Lab 2025.03) — 去掉 GRPO 的两个系统性偏置.

原论文《Understanding R1-Zero-Like Training: A Critical Perspective》(Liu et al.)
指出标准 GRPO 目标里藏着两个偏置，Dr.GRPO ("GRPO Done Right") 把它们都**去掉**：

    GRPO 目标:  (1/G) Σ_i  (1/|o_i|) Σ_t  min(ρ·Â, clip(ρ)·Â)
                其中       Â_i = (R_i − mean(R)) / std(R)

  ① 问题难度偏置 (question-level)：来自 advantage 里的 ` / std(R)`。
     reward 方差小的题（太简单 / 太难）会被放大权重。
     → Dr.GRPO **去掉 std 除法**：  Â_i = R_i − mean(R)   （只中心化，不缩放）

  ② 响应长度偏置 (response-level)：来自 loss 里的 ` 1/|o_i| `。
     长 response 的每 token 梯度被稀释；负 advantage 时短 response 反被多罚，
     净效果是鼓励"把错答案写长"。
     → Dr.GRPO **去掉 1/|o_i|**，改用常数归一（如 1 / (G·L_max)）。

关键：Dr.GRPO 是"做减法"（去归一化），**不是**换一种归一化、也**不是**加惩罚项。
本文件用两个对照演示分别坐实这两点。
"""
from __future__ import annotations

import torch


# ===== ① advantage 层：去掉 std 除法 =====

def grpo_advantage(rewards: torch.Tensor, k: int, eps: float = 1e-8) -> torch.Tensor:
    """标准 GRPO advantage：组内中心化 **再除以组 std**（含难度偏置）。"""
    R = rewards.reshape(-1, k)
    mean = R.mean(dim=1, keepdim=True)
    std = R.std(dim=1, keepdim=True) + eps
    A = (R - mean) / std
    return A.reshape(-1)


def dr_grpo_advantage(rewards: torch.Tensor, k: int) -> torch.Tensor:
    """Dr.GRPO advantage：**只组内中心化，不除 std**（去掉难度偏置）。

        Â_i = R_i − mean(R_group)
    """
    R = rewards.reshape(-1, k)
    A = R - R.mean(dim=1, keepdim=True)
    return A.reshape(-1)


# ===== ② loss 层：去掉 1/|o_i| 长度归一 =====

def grpo_length_weight(response_lens: torch.Tensor) -> torch.Tensor:
    """GRPO 把每条 response 的 loss 除以它自己的长度 |o_i| → 每条有效权重 1/|o_i|."""
    return 1.0 / response_lens.float().clamp(min=1)


def dr_grpo_length_weight(response_lens: torch.Tensor, l_const: float) -> torch.Tensor:
    """Dr.GRPO 用常数 L_max 归一 → 每条有效权重恒为 1/L_const（与长度无关，去偏）."""
    return torch.full_like(response_lens.float(), 1.0 / l_const)


# ===== 对照演示 =====

def compare_grpo_vs_dr() -> None:
    """对照①难度偏置：低方差组里 GRPO 把 advantage 放大，Dr.GRPO 不会."""
    print("[① 难度偏置] GRPO 除 std vs Dr.GRPO 仅中心化")
    # 组A reward 方差大；组B 方差小（题目"容易"，答案几乎都对）
    rewards = torch.tensor([0.0, 0.0, 1.0, 1.0,      # 组A: std 大
                            0.45, 0.5, 0.5, 0.55])   # 组B: std 小
    k = 4
    A_g = grpo_advantage(rewards, k)
    A_dr = dr_grpo_advantage(rewards, k)
    print(f"  rewards   : {rewards.tolist()}")
    print(f"  GRPO    A : {[round(a, 3) for a in A_g.tolist()]}")
    print(f"  Dr.GRPO A : {[round(a, 3) for a in A_dr.tolist()]}")
    print("  -> 组B(低方差)在 GRPO 里被 /std 放大到 ~±1.2(难度偏置)；Dr.GRPO 保持 ±0.05 原始尺度")


def compare_length_bias() -> None:
    """对照②长度偏置：GRPO 的 1/|o_i| 让长短 response 权重不同，Dr.GRPO 常数归一一致."""
    print("\n[② 长度偏置] GRPO 1/|o_i| vs Dr.GRPO 1/L_const")
    lens = torch.tensor([10.0, 50.0, 200.0])
    w_grpo = grpo_length_weight(lens)
    w_dr = dr_grpo_length_weight(lens, l_const=200.0)
    print(f"  response 长度       : {lens.tolist()}")
    print(f"  GRPO    每条权重 1/|o|   : {[round(w, 4) for w in w_grpo.tolist()]}")
    print(f"  Dr.GRPO 每条权重 1/L_max : {[round(w, 4) for w in w_dr.tolist()]}")
    print("  -> GRPO 短 response 权重高(长度偏置，鼓励写长错答案)；Dr.GRPO 所有长度同权，去偏")


if __name__ == "__main__":
    print("Dr. GRPO minimal\n" + "=" * 50)
    compare_grpo_vs_dr()
    compare_length_bias()
