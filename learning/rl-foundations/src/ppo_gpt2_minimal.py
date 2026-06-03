"""手写 PPO for GPT-2 — token-level PPO + KL ref penalty.

教学折衷：
    - 用 GPT-2-small (124M) 保证 5090 24GB 跑得动
    - critic 用独立 GPT-2-small + value head
    - reward 用句长（每多 1 字符 +0.05）的玩具版，便于看 reward 上升
    - 真实 RM 留给下一讲（用 BERT-sentiment）

⚠️ 由于 LLM-PPO 单 iter 30-60 s，本脚本默认 --total-iters 20，约 10-20 min。

运行:
    python learning/rl-foundations/src/ppo_gpt2_minimal.py
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


def length_reward(responses: list[str]) -> torch.Tensor:
    """玩具 reward：越长越好（每多 1 字符 +0.05），上限 5。"""
    return torch.tensor([min(len(r) * 0.05, 5.0) for r in responses])


def get_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """计算 input_ids[:, 1:] 的 log π。返回 (B, T-1)."""
    out = model(input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_pi = F.log_softmax(logits, dim=-1)
    return log_pi.gather(2, targets.unsqueeze(-1)).squeeze(-1)


class GPT2WithValueHead(nn.Module):
    """GPT-2 + 1 维 value head（独立 critic，避免 actor/critic 互相干扰）。"""

    def __init__(self, gpt2_model) -> None:
        super().__init__()
        self.gpt2 = gpt2_model
        hidden = gpt2_model.config.hidden_size
        self.v_head = nn.Linear(hidden, 1)
        nn.init.normal_(self.v_head.weight, std=0.01)
        nn.init.zeros_(self.v_head.bias)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None):
        out = self.gpt2(input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
        h = out.hidden_states[-1]              # (B, T, hidden)
        V = self.v_head(h).squeeze(-1)         # (B, T)
        return out.logits, V


def build_token_rewards(
    raw_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    log_p_act: torch.Tensor,
    log_p_ref: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """构造 token-level reward：仅末端给 raw_rewards，每 token 减 β · KL(act||ref).

    参数：
        raw_rewards: (B,) — RM 末端分数
        response_mask: (B, T-1) — 1 表示 response token，0 表示 prompt
        log_p_act, log_p_ref: (B, T-1)
        beta: KL 系数
    返回：
        rewards_per_token: (B, T-1)
    """
    kl = log_p_act - log_p_ref                  # per-token
    rewards = -beta * kl                        # 起手全是 KL penalty
    # 末端 token 位加上 raw_rewards
    last_idx = response_mask.long().sum(dim=1) - 1  # (B,) response 最后一 token
    for b in range(rewards.size(0)):
        rewards[b, last_idx[b]] += raw_rewards[b]
    rewards = rewards * response_mask           # 把 prompt 段的 reward 清零
    return rewards


def train(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"device = {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    actor_lm = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    critic_lm = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    critic = GPT2WithValueHead(critic_lm).to(device)
    ref_lm = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    ref_lm.eval()
    for p in ref_lm.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam(
        list(actor_lm.parameters()) + list(critic.parameters()),
        lr=args.lr,
    )

    # 玩具 prompts：50 条电影评论开头
    prompts = [
        "The movie was", "I think this film", "Honestly, this movie",
        "I just watched", "This is a", "What a great",
    ] * 10

    beta = args.beta
    for it in range(1, args.total_iters + 1):
        actor_lm.eval()
        critic.eval()

        # ---- Rollout ----
        batch_prompts = prompts[:args.batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
        prompt_len = enc.input_ids.size(1)

        with torch.no_grad():
            gen = actor_lm.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=True, top_p=0.9, temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_ids = gen
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        # 计算 reward
        responses = tokenizer.batch_decode(
            input_ids[:, prompt_len:], skip_special_tokens=True
        )
        raw_rewards = length_reward(responses).to(device)

        # ---- log_probs old + ref + values ----
        with torch.no_grad():
            log_p_old = get_log_probs(actor_lm, input_ids, attention_mask)
            log_p_ref = get_log_probs(ref_lm, input_ids, attention_mask)
            _, V_old = critic(input_ids, attention_mask=attention_mask)
            V_old = V_old[:, :-1]                # 与 log_p 对齐

        # response mask：prompt 段 0，response 段 1
        T_minus_1 = input_ids.size(1) - 1
        response_mask = torch.zeros_like(log_p_old)
        response_mask[:, prompt_len - 1:] = 1
        response_mask = response_mask * attention_mask[:, 1:].float()

        rewards = build_token_rewards(
            raw_rewards, response_mask, log_p_old, log_p_ref, beta=beta,
        )

        # ---- GAE token-level ----
        # 用每个 sample 自己的 T，简化：把整个 batch flatten 算 GAE 不严谨
        # 这里 sample-wise 处理
        from common import compute_gae
        adv_buf = torch.zeros_like(log_p_old)
        ret_buf = torch.zeros_like(log_p_old)
        for b in range(input_ids.size(0)):
            mask_b = response_mask[b].bool()
            if mask_b.sum() == 0:
                continue
            r_b = rewards[b][mask_b]
            V_b = V_old[b][mask_b]
            done_b = torch.zeros_like(r_b)
            done_b[-1] = 1.0
            A_b, R_b = compute_gae(
                r_b.unsqueeze(1), V_b.unsqueeze(1), done_b.unsqueeze(1),
                last_value=0.0, gamma=1.0, lam=args.lam,
            )
            adv_buf[b][mask_b] = A_b.squeeze(1)
            ret_buf[b][mask_b] = R_b.squeeze(1)

        # adv normalization
        valid = response_mask.bool()
        adv_buf[valid] = (adv_buf[valid] - adv_buf[valid].mean()) / (adv_buf[valid].std() + 1e-8)

        # ---- K epoch update ----
        actor_lm.train()
        critic.train()
        for epoch in range(args.K_epochs):
            log_p_new = get_log_probs(actor_lm, input_ids, attention_mask)
            _, V_new = critic(input_ids, attention_mask=attention_mask)
            V_new = V_new[:, :-1]

            ratio = (log_p_new - log_p_old).exp()
            surr1 = ratio * adv_buf
            surr2 = ratio.clamp(1 - args.eps, 1 + args.eps) * adv_buf
            L_clip = -(torch.min(surr1, surr2) * response_mask).sum() / response_mask.sum().clamp(min=1)

            L_vf = ((V_new - ret_buf) ** 2 * response_mask).sum() / response_mask.sum().clamp(min=1)
            loss = L_clip + args.vf_coef * L_vf

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(actor_lm.parameters()) + list(critic.parameters()),
                args.max_grad_norm,
            )
            opt.step()

        # ---- 监控 ----
        with torch.no_grad():
            log_p_now = get_log_probs(actor_lm, input_ids, attention_mask)
            kl_avg = ((log_p_now - log_p_ref) * response_mask).sum() / response_mask.sum().clamp(min=1)

        mean_R = raw_rewards.mean().item()
        avg_len = sum(len(r) for r in responses) / len(responses)
        print(f"Iter {it:3d} | mean raw_R {mean_R:6.3f} | mean len {avg_len:5.1f} | "
              f"L_clip {L_clip.item():7.4f} | L_vf {L_vf.item():7.4f} | "
              f"KL(act||ref) {kl_avg.item():6.3f} | β {beta:.3f}")

        # adaptive β
        if args.adaptive_kl:
            if kl_avg.item() > 1.5 * args.target_kl:
                beta = beta * 1.5
            elif kl_avg.item() < 0.5 * args.target_kl:
                beta = beta / 1.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--total-iters", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--K-epochs", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--target-kl", type=float, default=0.1)
    p.add_argument("--adaptive-kl", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
