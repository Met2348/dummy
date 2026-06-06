"""RL Foundations 通用工具函数。

复用 PEFT 系列的 freeze / 参数统计，并新增 RL 专用 helper：
- compute_returns / compute_gae / advantage
- log_prob 计算
- KL divergence 估计
- rollout 数据 batch 工具
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== PEFT 系列继承部分 =====

def freeze_base_model(model: nn.Module) -> None:
    """冻结所有参数（用于 LLM-RL 的 reference / critic 上）。"""
    for p in model.parameters():
        p.requires_grad = False


def print_param_summary(model: nn.Module, name: str = "model") -> None:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - train
    pct = 100 * train / total if total > 0 else 0
    print(f"\n  {name}")
    print(f"  Total params:      {total:>12,}")
    print(f"  Trainable params:  {train:>12,}  ({pct:.4f}%)")
    print(f"  Frozen params:     {frozen:>12,}")


# ===== RL 核心 helper =====

def compute_returns(
    rewards: list[float],
    dones: list[bool],
    gamma: float = 0.99,
) -> list[float]:
    """逆序 Bellman 计算 discounted return.

    G_t = r_t + γ G_{t+1}（done 时 G_{t+1}=0）。

    REINFORCE / Monte-Carlo policy gradient 用。
    """
    returns: list[float] = []
    G = 0.0
    for r, d in zip(reversed(rewards), reversed(dones)):
        if d:
            G = 0.0
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: float = 0.0,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation.

    δ_t = r_t + γ V(s_{t+1}) (1-d_t) - V(s_t)
    A_t = δ_t + γλ (1-d_t) A_{t+1}
    R_t = A_t + V(s_t)  (用于训 value head)

    返回 (advantages, returns)。两者 shape 同 rewards。
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0])
    next_value = torch.as_tensor(last_value, dtype=values.dtype, device=values.device)
    if next_value.ndim == 0 and values[0].ndim > 0:
        next_value = next_value.expand_as(values[0])
    for t in reversed(range(T)):
        mask = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def normalize_advantages(adv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """标准化 advantage：减均值除标准差，PPO 关键 trick 之一。"""
    return (adv - adv.mean()) / (adv.std() + eps)


# ===== 概率 / KL =====

def categorical_log_prob(
    logits: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """从 logits 计算所选动作的 log π(a|s).

    Args:
        logits: (B, num_actions)
        actions: (B,) long
    Returns:
        log_prob: (B,)
    """
    log_pi = F.log_softmax(logits, dim=-1)
    return log_pi.gather(1, actions.unsqueeze(1)).squeeze(1)


def categorical_entropy(logits: torch.Tensor) -> torch.Tensor:
    """H(π) = -Σ π log π，按 batch 平均。"""
    log_pi = F.log_softmax(logits, dim=-1)
    pi = log_pi.exp()
    return -(pi * log_pi).sum(-1).mean()


def kl_categorical(
    logits_old: torch.Tensor,
    logits_new: torch.Tensor,
) -> torch.Tensor:
    """KL(π_old || π_new) = Σ π_old (log π_old - log π_new)。"""
    log_old = F.log_softmax(logits_old, dim=-1)
    log_new = F.log_softmax(logits_new, dim=-1)
    return (log_old.exp() * (log_old - log_new)).sum(-1).mean()


def kl_approx_logp(log_p_old: torch.Tensor, log_p_new: torch.Tensor) -> torch.Tensor:
    """Schulman approx KL — 不需要 full distribution。

    KL ≈ E[r - 1 - log r], 其中 r = π_new/π_old = exp(log_p_new - log_p_old)
    （http://joschu.net/blog/kl-approx.html）。
    """
    log_r = log_p_new - log_p_old
    r = log_r.exp()
    return (r - 1 - log_r).mean()


# ===== 训练辅助 =====

def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """1 - Var(y_true - y_pred) / Var(y_true)。

    1 = 完美预测，0 = 与均值相当，<0 = 比均值还差。
    用于诊断 critic 学得好不好。
    """
    var_y = y_true.var().item()
    if var_y == 0:
        return 0.0
    return 1.0 - (y_true - y_pred).var().item() / var_y


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    """[r0, r1, r2] → [r0+γr1+γ²r2, r1+γr2, r2]，纯 numpy 版本。"""
    out = np.zeros_like(x, dtype=np.float64)
    running = 0.0
    for i in reversed(range(len(x))):
        running = x[i] + discount * running
        out[i] = running
    return out


# ===== batch / rollout 工具 =====

class RolloutBuffer:
    """简单 on-policy rollout buffer.

    存:  obs / actions / log_probs / values / rewards / dones
    用法:
        buf = RolloutBuffer()
        for _ in range(T):
            buf.add(obs, action, log_prob, value, reward, done)
        adv, ret = buf.compute_gae(last_value, gamma, lam)
        batches = buf.get_minibatches(batch_size, n_epoch)
    """

    def __init__(self) -> None:
        self.obs: list = []
        self.actions: list = []
        self.log_probs: list = []
        self.values: list = []
        self.rewards: list = []
        self.dones: list = []

    def add(self, obs, action, log_prob, value, reward, done) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self) -> None:
        for lst in (self.obs, self.actions, self.log_probs,
                    self.values, self.rewards, self.dones):
            lst.clear()

    def compute_gae(self, last_value: float, gamma: float = 0.99,
                    lam: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.tensor(self.values, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)
        return compute_gae(rewards, values, dones, last_value, gamma, lam)


def to_tensor(x, device: str = "cpu") -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


# ===== 训练种子 =====

def set_seed(seed: int) -> None:
    """统一设种子，保证算法一致性测试可重复。"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
