"""Capstone: GPT-2 + IMDb + DistilBERT-SST2 RM + PPO.

完整三段 pipeline:
    1. Dataset: IMDb prompts (取前 120 字符)
    2. Rollout: GPT-2 续写 max_new_tokens
    3. RM: distilbert SST-2 当 sentiment scorer
    4. PPO: 优先 trl PPOTrainer（经典 API）；不可用时**自动回退**到手写
       token-level PPO（复用 ppo_gpt2_minimal 的实现）+ 真实 SST-2 RM。

为何要回退：trl 0.12+ 移除/重构了经典 `PPOConfig`/`PPOTrainer` 情感微调 API，
本仓库环境装的是 trl 1.5.x。回退路径保证"照 README 敲就能在 3080 Ti 上真跑通"，
而不是静默 no-op（假成功）。教学上手写版与 trl 等价，且复用你自己写的 PPO。

预期：full 跑（--total-iters 30+）mean_R(pos-prob) 上升；smoke（--total-iters 2）仅验证可跑通。

运行：
    python learning/rl-foundations/src/capstone_imdb_ppo.py
    python learning/rl-foundations/src/capstone_imdb_ppo.py --total-iters 30 --batch-size 4
    # 快速 smoke（验证可跑通，不保证提升）：
    python learning/rl-foundations/src/capstone_imdb_ppo.py --total-iters 2 --batch-size 4 --n-prompts 32
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


def load_prompts(n: int = 1000) -> list[str]:
    """从 IMDb 取前 n 条评论开头 120 字符当 prompt；离线/不可用时回退内置小样本。

    注：datasets 4.x+ 移除了裸别名 "imdb"，须用命名空间 id "stanfordnlp/imdb"。
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("stanfordnlp/imdb", split=f"train[:{n}]")
        return [t[:120] for t in ds["text"]]
    except Exception as exc:  # 离线 / Hub 变更 / 限流 → 不让 capstone 卡死
        print(f"[capstone] IMDb 加载失败（{type(exc).__name__}: {exc}）；回退内置 prompts。")
        base = [
            "The movie was", "I think this film", "Honestly, this movie",
            "I just watched", "This is a", "What a great", "The plot of",
            "After seeing it,", "My honest opinion is", "The acting in this",
        ]
        return [base[i % len(base)] for i in range(n)]


def train(args):
    """Dispatcher：经典 trl PPOTrainer 可用则用之，否则回退手写 PPO + 真实 RM。"""
    try:
        from trl import (  # noqa: F401
            PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead,
        )
        have_trl_ppo = True
    except ImportError as exc:
        print(f"[capstone] 经典 trl PPOTrainer API 不可用：{exc}")
        print("[capstone] -> 回退到手写 token-level PPO + 真实 SST-2 RM"
              "（教学等价，3080 Ti 可跑）。")
        have_trl_ppo = False

    if have_trl_ppo:
        train_trl(args)
    else:
        train_handwritten(args)


def train_trl(args):
    """原始路径：trl 经典 PPOTrainer（需 trl<0.12）。"""
    from transformers import AutoTokenizer
    from trl import (
        PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead,
    )
    from sentiment_reward import SentimentReward

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"device = {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading actor + ref + value head...")
    actor = AutoModelForCausalLMWithValueHead.from_pretrained(args.model)
    ref = AutoModelForCausalLMWithValueHead.from_pretrained(args.model)
    if device == "cuda":
        actor.to(device)
        ref.to(device)

    print("Loading SST-2 RM...")
    rm = SentimentReward(model_name=args.rm_model, device=device,
                         dtype=torch.float16 if device == "cuda" else torch.float32)

    print("Loading IMDb prompts...")
    prompts_text = load_prompts(args.n_prompts)
    prompts_tokens = [
        tokenizer(p, return_tensors="pt", truncation=True,
                  max_length=30).input_ids[0].to(device)
        for p in prompts_text
    ]

    config = PPOConfig(
        model_name=args.model,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.K_epochs,
        cliprange=args.eps,
        vf_coef=args.vf_coef,
        init_kl_coef=args.beta,
        target_kl=args.target_kl,
        adap_kl_ctrl=True,
        log_with=None,
    )
    trainer = PPOTrainer(config, actor, ref, tokenizer)

    initial_R = None
    mean_R = 0.0
    for it in range(1, args.total_iters + 1):
        start = ((it - 1) * args.batch_size) % len(prompts_tokens)
        batch_prompts = prompts_tokens[start:start + args.batch_size]
        if len(batch_prompts) < args.batch_size:
            batch_prompts = (
                batch_prompts + prompts_tokens[: args.batch_size - len(batch_prompts)]
            )

        response_tensors = trainer.generate(
            batch_prompts,
            max_new_tokens=args.max_new_tokens,
            do_sample=True, top_p=0.9, temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        responses_text = [
            tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors
        ]
        full_texts = [
            tokenizer.decode(p, skip_special_tokens=True) + r
            for p, r in zip(batch_prompts, responses_text)
        ]

        rewards = rm.score(full_texts)
        reward_list = [r.unsqueeze(0).to(device) for r in rewards]

        trainer.step(batch_prompts, response_tensors, reward_list)

        mean_R = rewards.mean().item()
        if initial_R is None:
            initial_R = mean_R
        print(f"Iter {it:3d} | mean_R {mean_R:.3f}")

    _report(initial_R, mean_R)


def train_handwritten(args):
    """回退路径：复用 ppo_gpt2_minimal 的 token-level PPO + 真实 SST-2 RM。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ppo_gpt2_minimal import (
        GPT2WithValueHead, build_token_rewards, get_log_probs,
    )
    from common import compute_gae
    from sentiment_reward import SentimentReward

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"device = {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # decoder-only 批量 generate 须左填充

    print("Loading actor + critic + ref (GPT-2)...")
    actor_lm = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    critic = GPT2WithValueHead(
        AutoModelForCausalLM.from_pretrained(args.model)
    ).to(device)
    ref_lm = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    ref_lm.eval()
    for p in ref_lm.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam(
        list(actor_lm.parameters()) + list(critic.parameters()), lr=args.lr,
    )

    print("Loading SST-2 RM...")
    rm = SentimentReward(model_name=args.rm_model, device=device,
                         dtype=torch.float16 if device == "cuda" else torch.float32)

    print("Loading IMDb prompts...")
    prompts = load_prompts(args.n_prompts)

    beta = args.beta
    initial_R = None
    mean_R = 0.0
    for it in range(1, args.total_iters + 1):
        actor_lm.eval()
        critic.eval()

        start = ((it - 1) * args.batch_size) % max(len(prompts), 1)
        batch_prompts = prompts[start:start + args.batch_size]
        if len(batch_prompts) < args.batch_size:
            batch_prompts = batch_prompts + prompts[: args.batch_size - len(batch_prompts)]

        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=40).to(device)
        prompt_len = enc.input_ids.size(1)
        with torch.no_grad():
            gen = actor_lm.generate(
                **enc, max_new_tokens=args.max_new_tokens,
                do_sample=True, top_p=0.9, temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_ids = gen
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        full_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        raw_rewards = rm.score(full_texts).to(device)

        with torch.no_grad():
            log_p_old = get_log_probs(actor_lm, input_ids, attention_mask)
            log_p_ref = get_log_probs(ref_lm, input_ids, attention_mask)
            _, V_old = critic(input_ids, attention_mask=attention_mask)
            V_old = V_old[:, :-1]

        response_mask = torch.zeros_like(log_p_old)
        response_mask[:, prompt_len - 1:] = 1
        response_mask = response_mask * attention_mask[:, 1:].float()

        rewards = build_token_rewards(
            raw_rewards, response_mask, log_p_old, log_p_ref, beta=beta,
        )

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
                last_value=0.0, gamma=1.0, lam=0.95,
            )
            adv_buf[b][mask_b] = A_b.squeeze(1)
            ret_buf[b][mask_b] = R_b.squeeze(1)
        valid = response_mask.bool()
        if valid.any():
            adv_buf[valid] = (adv_buf[valid] - adv_buf[valid].mean()) / (adv_buf[valid].std() + 1e-8)

        actor_lm.train()
        critic.train()
        for _ in range(args.K_epochs):
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
                list(actor_lm.parameters()) + list(critic.parameters()), 1.0,
            )
            opt.step()

        mean_R = raw_rewards.mean().item()
        if initial_R is None:
            initial_R = mean_R
        print(f"Iter {it:3d} | mean_R(pos-prob) {mean_R:.3f} | beta {beta:.3f}")

    _report(initial_R, mean_R)


def _report(initial_R, final_R):
    if initial_R is None:
        return
    delta = (final_R - initial_R) / max(abs(initial_R), 0.01) * 100
    print(f"\n==> initial mean_R = {initial_R:.3f}")
    print(f"==> final   mean_R = {final_R:.3f}")
    print(f"==> 提升 = {delta:+.1f}%  (smoke 迭代太少时不保证上升，仅验证可跑通)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")    # gpt2-medium 需更大显存；默认 small
    p.add_argument("--rm-model", default="distilbert-base-uncased-finetuned-sst-2-english")
    p.add_argument("--n-prompts", type=int, default=1000)
    p.add_argument("--total-iters", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--mini-batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--lr", type=float, default=1.41e-5)
    p.add_argument("--K-epochs", type=int, default=4)
    p.add_argument("--eps", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--target-kl", type=float, default=6.0)
    p.add_argument("--tb-log-dir", type=Path, default=Path("runs/capstone_imdb"))
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
