"""PPO for LLM minimal — InstructGPT 第三阶段.

4 model:
    actor  — 训练的 policy
    critic — value head (与 actor backbone 共享，从 RM 初始化)
    ref    — 冻结的 SFT model (KL 约束)
    rm     — 冻结的 reward model (终态打分)

继承 rl-foundations/ppo_gpt2_minimal.py 的 token-level reward 构造.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 继承 rl-foundations
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "learning" / "rl-foundations" / "src"))
try:
    from common import compute_gae  # noqa: E402
except ImportError:
    def compute_gae(rewards, values, dones, last_value=0.0, gamma=0.99, lam=0.95):
        T = rewards.shape[0]
        adv = torch.zeros_like(rewards)
        gae, next_v = 0.0, last_value
        for t in reversed(range(T)):
            mask = 1.0 - dones[t].float()
            delta = rewards[t] + gamma * next_v * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            adv[t] = gae
            next_v = values[t].item()
        return adv, adv + values


class ActorCritic(nn.Module):
    """共享 backbone 的 actor + critic。"""

    def __init__(self, lm_model):
        super().__init__()
        self.lm = lm_model
        hidden = lm_model.config.hidden_size if hasattr(lm_model.config, "hidden_size") \
            else lm_model.config.n_embd
        self.v_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask=None):
        out = self.lm(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        h = out.hidden_states[-1]
        values = self.v_head(h).squeeze(-1)
        return out.logits, values


def gather_logp(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """取实际 token 的 log p."""
    log_probs = F.log_softmax(logits, dim=-1)
    # shift: prediction at t 对应 token t+1
    log_p = log_probs[:, :-1].gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return log_p


def build_token_rewards(
    raw_rewards: torch.Tensor,   # (B,) RM 终态分数
    response_mask: torch.Tensor, # (B, T)
    log_p_act: torch.Tensor,     # (B, T)
    log_p_ref: torch.Tensor,     # (B, T)
    beta: float = 0.02,
) -> torch.Tensor:
    """token-level reward = -β·KL_t + δ(t=last)·RM."""
    kl = log_p_act - log_p_ref          # KL approx
    rewards = -beta * kl                # 每步 KL 罚
    last_idx = response_mask.long().sum(dim=1) - 1
    for b in range(rewards.size(0)):
        rewards[b, last_idx[b]] += raw_rewards[b]
    return rewards * response_mask


def ppo_step(
    actor_critic, ref_model, batch, optimizer,
    clip_eps: float = 0.2, vf_coef: float = 0.5, ent_coef: float = 0.01,
):
    """单步 PPO 更新（actor + critic 联合）."""
    input_ids = batch["input_ids"]
    mask = batch["response_mask"]
    advantages = batch["advantages"]
    returns = batch["returns"]
    log_p_old = batch["log_p_old"]

    logits, values = actor_critic(input_ids)
    log_p_new = gather_logp(logits, input_ids)

    # PPO clip
    ratio = (log_p_new - log_p_old).exp()
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
    pi_loss = -torch.min(surr1, surr2)
    pi_loss = (pi_loss * mask[:, 1:]).sum() / mask[:, 1:].sum().clamp(min=1)

    # value loss
    v = values[:, 1:]
    v_loss = ((v - returns) ** 2 * mask[:, 1:]).sum() / mask[:, 1:].sum().clamp(min=1)

    # entropy bonus
    probs = F.softmax(logits[:, :-1], dim=-1)
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    entropy = -(probs * log_probs).sum(-1)
    ent_loss = (entropy * mask[:, 1:]).sum() / mask[:, 1:].sum().clamp(min=1)

    total = pi_loss + vf_coef * v_loss - ent_coef * ent_loss
    optimizer.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 1.0)
    optimizer.step()

    return {
        "pi_loss": pi_loss.item(),
        "v_loss": v_loss.item(),
        "entropy": ent_loss.item(),
        "total": total.item(),
    }


if __name__ == "__main__":
    print("PPO-LLM minimal — smoke test")
    torch.manual_seed(0)
    B, T, V, H = 2, 12, 100, 32

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("C", (), {"hidden_size": H})()
            self.emb = nn.Embedding(V, H)
            self.lm_head = nn.Linear(H, V)

        def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
            h = self.emb(input_ids)
            return type("O", (), {
                "logits": self.lm_head(h),
                "hidden_states": (h,),
            })()

    ac = ActorCritic(TinyLM())
    ref = ActorCritic(TinyLM())
    for p in ref.parameters():
        p.requires_grad = False

    input_ids = torch.randint(0, V, (B, T))
    response_mask = torch.cat([torch.zeros(B, 4), torch.ones(B, T - 4)], dim=1)

    with torch.no_grad():
        logits_old, values_old = ac(input_ids)
        log_p_old = gather_logp(logits_old, input_ids)
        logits_ref, _ = ref(input_ids)
        log_p_ref = gather_logp(logits_ref, input_ids)

    raw_rewards = torch.tensor([1.0, -0.5])
    token_rewards = build_token_rewards(raw_rewards, response_mask[:, 1:], log_p_old, log_p_ref, beta=0.02)

    dones = torch.zeros_like(token_rewards)
    dones[:, -1] = 1.0
    adv, ret = [], []
    for b in range(B):
        a, r = compute_gae(token_rewards[b], values_old[b, 1:], dones[b])
        adv.append(a); ret.append(r)
    advantages = torch.stack(adv)
    returns = torch.stack(ret)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    optim = torch.optim.AdamW(ac.parameters(), lr=1e-4)
    batch = dict(
        input_ids=input_ids, response_mask=response_mask,
        advantages=advantages, returns=returns, log_p_old=log_p_old,
    )
    for step in range(3):
        stats = ppo_step(ac, ref, batch, optim)
        print(f"step {step}: {stats}")
