"""Capstone: GPT-2-medium + IMDb + BERT-SST2 RM + trl PPOTrainer.

完整三段 pipeline:
    1. Dataset: IMDb 1k prompts (取前 120 字符)
    2. Rollout: GPT-2-medium 续写 30 token
    3. RM: distilbert SST-2 当 sentiment scorer
    4. PPO: trl PPOTrainer

预期：256 iter (~4-6h on 5090 24GB), mean_reward 0.45 → 0.6+。

运行：
    python learning/rl-foundations/src/capstone_imdb_ppo.py
    python learning/rl-foundations/src/capstone_imdb_ppo.py --total-iters 30 \\
        --batch-size 4   # 快速 smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def load_prompts(n: int = 1000) -> list[str]:
    """从 IMDb 取前 n 条评论的开头 120 字符当 prompt。"""
    from datasets import load_dataset
    ds = load_dataset("imdb", split=f"train[:{n}]")
    return [t[:120] for t in ds["text"]]


def train(args):
    from transformers import AutoTokenizer
    try:
        from trl import (
            PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead,
        )
    except ImportError as e:
        print(f"trl 未安装: {e}")
        return

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

    log_dir = args.tb_log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir)
    except ImportError:
        print("tensorboard 不可用，跳过日志")

    initial_R = None
    for it in range(1, args.total_iters + 1):
        # 取一个 batch 的 prompts
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

        stats = trainer.step(batch_prompts, response_tensors, reward_list)

        mean_R = rewards.mean().item()
        if initial_R is None:
            initial_R = mean_R
        kl = stats.get("objective/kl", 0)
        avg_len = sum(len(r) for r in responses_text) / len(responses_text)

        print(f"Iter {it:3d} | mean_R {mean_R:.3f} | KL {kl:.3f} | "
              f"len {avg_len:.1f}")

        if writer:
            writer.add_scalar("ppo/mean_reward", mean_R, it)
            writer.add_scalar("ppo/kl", kl, it)
            writer.add_scalar("ppo/mean_response_len", avg_len, it)

        # Every 50 iter，spot check 1 个样本
        if it % 50 == 0:
            print(f"  [spot]: {full_texts[0][:150]}")

    if initial_R is not None:
        final_R = mean_R
        delta = (final_R - initial_R) / max(abs(initial_R), 0.01) * 100
        print(f"\n==> initial mean_R = {initial_R:.3f}")
        print(f"==> final mean_R   = {final_R:.3f}")
        print(f"==> 提升 = {delta:+.1f}%")
        passed = delta >= 30
        print(f"\nCAPSTONE {'PASSED ✅' if passed else 'NOT PASSED ⚠️'}")

    if writer:
        writer.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")    # gpt2-medium 需 24GB；默认 small
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
