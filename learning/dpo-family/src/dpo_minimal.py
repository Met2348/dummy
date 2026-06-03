"""手写 DPO loss — 4 行核心实现.

DPO loss = -log sigmoid( β · (log_ratio_chosen - log_ratio_rejected) )

其中 log_ratio = log π_θ(y|x) - log π_ref(y|x)

运行：
    python learning/dpo-family/src/dpo_minimal.py
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_log_probs_for_labels(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """计算 response 段 token 的 log π 总和（per sample）。

    args:
        input_ids:        (B, T)
        attention_mask:   (B, T)
        response_mask:    (B, T-1) — 1 表示 response 位
    returns:
        per-sample log p (B,)
    """
    out = model(input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]                       # 预测下一 token
    targets = input_ids[:, 1:]                            # 真实下一 token
    log_p_all = F.log_softmax(logits, dim=-1)
    log_p_taken = log_p_all.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    # masking
    log_p_taken = log_p_taken * response_mask
    return log_p_taken.sum(-1)                            # 每条 sample 求和


def dpo_loss(
    log_p_chosen_actor: torch.Tensor,
    log_p_chosen_ref: torch.Tensor,
    log_p_rejected_actor: torch.Tensor,
    log_p_rejected_ref: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """DPO loss = -log sigmoid(β · (log_ratio_w - log_ratio_l))."""
    log_ratio_w = log_p_chosen_actor - log_p_chosen_ref
    log_ratio_l = log_p_rejected_actor - log_p_rejected_ref
    margin = beta * (log_ratio_w - log_ratio_l)
    return -F.logsigmoid(margin).mean()


def train(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    actor = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    ref = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(actor.parameters(), lr=args.lr)

    try:
        ds = load_dataset("Anthropic/hh-rlhf", split=f"train[:{args.n_train}]")
        ds = list(ds)
    except Exception:
        print("⚠️  HH 加载失败，用 dummy")
        ds = [{"chosen": "answer A", "rejected": "answer B"}
              for _ in range(args.n_train)]

    def tokenize_one(text, max_len=args.max_length):
        return tokenizer(text, truncation=True, max_length=max_len,
                         return_tensors="pt")

    for epoch in range(args.epochs):
        tot_loss, tot_margin, n = 0.0, 0.0, 0
        for ex in ds:
            chosen = tokenize_one(ex["chosen"]).to(device)
            rejected = tokenize_one(ex["rejected"]).to(device)

            # response_mask = 全部 1（dummy；真实场景应只 mask response 段）
            resp_mask_c = torch.ones_like(chosen.input_ids[:, 1:], dtype=torch.float32)
            resp_mask_r = torch.ones_like(rejected.input_ids[:, 1:], dtype=torch.float32)

            log_p_c_act = get_log_probs_for_labels(
                actor, chosen.input_ids, chosen.attention_mask, resp_mask_c
            )
            log_p_r_act = get_log_probs_for_labels(
                actor, rejected.input_ids, rejected.attention_mask, resp_mask_r
            )
            with torch.no_grad():
                log_p_c_ref = get_log_probs_for_labels(
                    ref, chosen.input_ids, chosen.attention_mask, resp_mask_c
                )
                log_p_r_ref = get_log_probs_for_labels(
                    ref, rejected.input_ids, rejected.attention_mask, resp_mask_r
                )

            loss = dpo_loss(log_p_c_act, log_p_c_ref, log_p_r_act, log_p_r_ref,
                            beta=args.beta)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()

            margin = ((log_p_c_act - log_p_c_ref) - (log_p_r_act - log_p_r_ref)).mean()
            tot_loss += loss.item()
            tot_margin += margin.item()
            n += 1

            if n % args.log_interval == 0:
                print(f"Step {n:5d} | loss {tot_loss/n:.4f} | "
                      f"margin {tot_margin/n:+.3f}")

        print(f"== Epoch {epoch + 1} done ==")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
