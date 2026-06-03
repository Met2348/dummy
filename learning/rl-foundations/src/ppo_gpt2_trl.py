"""trl PPOTrainer 对照 — GPT-2 + 长度 reward.

教学目标：trl 的 PPOTrainer 自动处理 4 model（actor / ref / critic via value head / RM）
显存细节，是 LLM-PPO 的工业标准接口。

⚠️ trl 0.13 后接口稳定但仍可能 API 微调，本脚本以 trl 0.13 为准。

运行:
    python learning/rl-foundations/src/ppo_gpt2_trl.py --total-iters 20
"""
from __future__ import annotations

import argparse

import torch


def length_reward(responses: list[str]) -> torch.Tensor:
    return torch.tensor([min(len(r) * 0.05, 5.0) for r in responses])


def train(args):
    from transformers import AutoTokenizer
    try:
        from trl import (
            PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
        )
    except ImportError as e:
        print(f"trl 未安装或版本不兼容: {e}")
        print("pip install trl>=0.13")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model)
    ref = AutoModelForCausalLMWithValueHead.from_pretrained(args.model)

    config = PPOConfig(
        model_name=args.model,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.batch_size,
        ppo_epochs=args.K_epochs,
        cliprange=args.eps,
        vf_coef=args.vf_coef,
        init_kl_coef=args.beta,
        target_kl=args.target_kl,
        adap_kl_ctrl=args.adaptive_kl,
        log_with=None,
    )
    trainer = PPOTrainer(config, model, ref, tokenizer)

    prompts_text = [
        "The movie was", "I think this film", "Honestly, this movie",
        "I just watched", "This is a", "What a great",
    ] * 10
    prompt_tensors = [
        tokenizer(p, return_tensors="pt").input_ids[0]
        for p in prompts_text[:args.batch_size]
    ]

    for it in range(1, args.total_iters + 1):
        # Generate
        response_tensors = trainer.generate(
            prompt_tensors,
            max_new_tokens=args.max_new_tokens,
            do_sample=True, top_p=0.9, temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        responses = [
            tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors
        ]

        # Reward
        raw_rewards = length_reward(responses)
        reward_list = [r.unsqueeze(0) for r in raw_rewards]

        stats = trainer.step(prompt_tensors, response_tensors, reward_list)

        mean_R = raw_rewards.mean().item()
        avg_len = sum(len(r) for r in responses) / len(responses)
        print(f"Iter {it:3d} | mean raw_R {mean_R:6.3f} | mean len {avg_len:5.1f} | "
              f"kl {stats.get('objective/kl', 0):.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--total-iters", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--lr", type=float, default=1.41e-5)
    p.add_argument("--eps", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.1)
    p.add_argument("--K-epochs", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--target-kl", type=float, default=6.0)
    p.add_argument("--adaptive-kl", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
