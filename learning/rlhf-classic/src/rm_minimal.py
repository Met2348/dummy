"""手写 Reward Model 训练 — Bradley-Terry pairwise loss.

教学目标：
    1. RM = LM + scalar head（取最后非 pad token 的 hidden）
    2. BT loss = -log sigmoid(r_chosen - r_rejected)
    3. 在 Anthropic-HH 子集上跑 1 epoch

运行：
    python learning/rlhf-classic/src/rm_minimal.py
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    """LM + 1 维 scalar head；取最后非 pad token 的 hidden 当 reward。"""

    def __init__(self, base_lm) -> None:
        super().__init__()
        self.lm = base_lm
        hidden = base_lm.config.hidden_size
        self.v_head = nn.Linear(hidden, 1)
        nn.init.normal_(self.v_head.weight, std=0.01)
        nn.init.zeros_(self.v_head.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """返回 (B,) 维 reward."""
        out = self.lm(input_ids, attention_mask=attention_mask,
                      output_hidden_states=True)
        h = out.hidden_states[-1]                     # (B, T, hidden)
        last_idx = attention_mask.sum(-1) - 1          # (B,)
        last_h = h[torch.arange(len(h)), last_idx]    # (B, hidden)
        r = self.v_head(last_h).squeeze(-1)            # (B,)
        return r


def bt_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(r_chosen - r_rejected).mean()


def accuracy(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> float:
    return (r_chosen > r_rejected).float().mean().item()


def train(args):
    from transformers import AutoModel, AutoTokenizer
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModel.from_pretrained(args.model).to(device)
    model = RewardModel(base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # 加载 Anthropic-HH
    try:
        ds = load_dataset("Anthropic/hh-rlhf", split=f"train[:{args.n_train}]")
        ds_eval = load_dataset("Anthropic/hh-rlhf", split=f"test[:{args.n_eval}]")
    except Exception as e:
        print(f"⚠️  Anthropic-HH 加载失败: {e}")
        print("    用 dummy 数据演示")
        ds = [
            {"chosen": "This is a great answer.", "rejected": "bad"}
            for _ in range(args.n_train)
        ]
        ds_eval = ds[:args.n_eval]

    def collate(batch):
        chosen = tokenizer([b["chosen"] for b in batch], return_tensors="pt",
                           padding=True, truncation=True, max_length=args.max_length)
        rejected = tokenizer([b["rejected"] for b in batch], return_tensors="pt",
                             padding=True, truncation=True, max_length=args.max_length)
        return {
            "chosen_ids": chosen.input_ids.to(device),
            "chosen_mask": chosen.attention_mask.to(device),
            "rejected_ids": rejected.input_ids.to(device),
            "rejected_mask": rejected.attention_mask.to(device),
        }

    from torch.utils.data import DataLoader
    train_loader = DataLoader(list(ds), batch_size=args.batch, shuffle=True,
                              collate_fn=collate)
    eval_loader = DataLoader(list(ds_eval), batch_size=args.batch,
                             collate_fn=collate)

    for epoch in range(args.epochs):
        model.train()
        tot_loss, tot_acc, n_batches = 0.0, 0.0, 0
        for batch in train_loader:
            r_c = model(batch["chosen_ids"], batch["chosen_mask"])
            r_r = model(batch["rejected_ids"], batch["rejected_mask"])
            loss = bt_loss(r_c, r_r)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item()
            tot_acc += accuracy(r_c, r_r)
            n_batches += 1
        print(f"Epoch {epoch+1} | train loss {tot_loss/n_batches:.4f} | "
              f"train acc {tot_acc/n_batches:.3f}")

        model.eval()
        eval_acc = 0.0
        n_eval = 0
        with torch.no_grad():
            for batch in eval_loader:
                r_c = model(batch["chosen_ids"], batch["chosen_mask"])
                r_r = model(batch["rejected_ids"], batch["rejected_mask"])
                eval_acc += accuracy(r_c, r_r)
                n_eval += 1
        print(f"  → eval acc {eval_acc/max(n_eval,1):.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
