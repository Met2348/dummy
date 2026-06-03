"""SFT (Supervised Fine-Tuning) minimal — InstructGPT 第一阶段.

NLL loss on response tokens only (mask prompt).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    """简化版 SFT 数据：每条 = (prompt_ids, response_ids)，拼接后只对 response 段计 loss."""

    def __init__(self, samples, tokenizer, max_len: int = 512):
        self.samples = samples
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, response = self.samples[idx]
        p_ids = self.tok.encode(prompt, add_special_tokens=False)
        r_ids = self.tok.encode(response, add_special_tokens=False) + [self.tok.eos_token_id]
        input_ids = (p_ids + r_ids)[: self.max_len]
        # labels: -100 在 prompt 段（不计 loss），response 段为本身
        labels = [-100] * min(len(p_ids), self.max_len)
        labels += r_ids[: self.max_len - len(labels)]
        # pad
        pad_len = self.max_len - len(input_ids)
        input_ids = input_ids + [self.tok.pad_token_id] * pad_len
        labels = labels + [-100] * pad_len
        return torch.tensor(input_ids), torch.tensor(labels)


def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """标准 next-token NLL，labels = -100 处忽略."""
    # shift
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


def train_sft_step(model, batch, optimizer):
    """单步 SFT 训练."""
    input_ids, labels = batch
    out = model(input_ids=input_ids)
    loss = sft_loss(out.logits, labels)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


if __name__ == "__main__":
    print("SFT minimal — 演示")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    samples = [
        ("Q: 1+1=? A:", " 2"),
        ("Q: capital of France? A:", " Paris"),
    ]
    ds = SFTDataset(samples, tok, max_len=32)
    x, y = ds[0]
    print(f"input_ids shape: {x.shape}")
    print(f"label mask (-100 表 prompt): {(y == -100).sum().item()}/{len(y)}")

    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for step in range(3):
        loss = train_sft_step(model, (x.unsqueeze(0), y.unsqueeze(0)), optim)
        print(f"step {step} loss={loss:.4f}")
