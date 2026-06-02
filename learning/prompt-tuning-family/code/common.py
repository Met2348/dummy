"""
共用工具（全专题共享）：
- 玩具任务数据集（情感二分类）
- 参数量统计
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset


POSITIVE_SAMPLES = [
    "I love this movie",
    "This is amazing",
    "Best day ever",
    "What a fantastic show",
    "Absolutely wonderful",
]

NEGATIVE_SAMPLES = [
    "I hate this",
    "This is terrible",
    "Worst day ever",
    "What a boring show",
    "Absolutely awful",
]


class ToySentimentDataset(Dataset):
    """玩具情感数据集，仅用于演示训练流程。

    每条样本：input_ids (max_len,), attention_mask (max_len,), label (scalar)。
    """

    def __init__(self, tokenizer, max_len: int = 16):
        self.samples = [(s, 1) for s in POSITIVE_SAMPLES] + [(s, 0) for s in NEGATIVE_SAMPLES]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        text, label = self.samples[idx]
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """返回 (可训练参数量, 总参数量)。"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def print_param_summary(model: torch.nn.Module, name: str = "model") -> None:
    trainable, total = count_parameters(model)
    pct = 100.0 * trainable / total if total else 0.0
    print(f"[{name}] 可训练参数={trainable:,} / 总参数={total:,} ({pct:.4f}%)")


if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    ds = ToySentimentDataset(tok)
    print(f"数据集大小: {len(ds)}")
    print(f"样本 0: input_ids.shape={ds[0]['input_ids'].shape}, label={ds[0]['label']}")
