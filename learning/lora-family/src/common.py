"""LoRA 家族专题共享工具。

复用 prompt-tuning-family 的工具 + 新增 LoRA 相关 utility。
"""
from __future__ import annotations

import random
from typing import Iterable

import torch
import torch.nn as nn


# ==============================
# 通用参数统计
# ==============================
def count_parameters(model: nn.Module) -> tuple[int, int]:
    """返回 (trainable, total)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def print_param_summary(model: nn.Module, name: str = "model") -> None:
    trainable, total = count_parameters(model)
    pct = 100 * trainable / total if total else 0.0
    print(f"  {name}")
    print(f"  Total params:     {total:>12,}")
    print(f"  Trainable params: {trainable:>12,}  ({pct:.4f}%)")
    print(f"  Frozen params:    {total - trainable:>12,}")


# ==============================
# 设备工具
# ==============================
def device_auto() -> torch.device:
    """优先 GPU，回退 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================
# LoRA / PEFT 通用工具
# ==============================
def freeze_base_model(model: nn.Module) -> None:
    """把所有参数 requires_grad=False。"""
    for p in model.parameters():
        p.requires_grad = False


def target_linear_modules(
    model: nn.Module,
    target_names: tuple[str, ...] = ("c_attn",),
) -> list[tuple[str, nn.Module]]:
    """找到名字结尾匹配 target_names 的 nn.Linear / GPT-2 Conv1D 模块。

    Returns:
        [(qualified_name, module), ...]
    """
    matches = []
    for name, module in model.named_modules():
        if any(name.endswith(t) for t in target_names):
            matches.append((name, module))
    return matches


def get_parent_and_attr(root: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    """根据 'h.0.attn.c_attn' 找到 parent module 和 attr name，方便 setattr 替换。"""
    parts = qualified_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    return parent, parts[-1]


def replace_modules(
    model: nn.Module,
    target_names: tuple[str, ...],
    factory: callable,
) -> None:
    """对 model 中所有名字结尾匹配 target_names 的模块，用 factory(old_module) 替换。

    factory: (old_module) -> new_module
    """
    matches = target_linear_modules(model, target_names)
    for qname, old in matches:
        parent, attr = get_parent_and_attr(model, qname)
        setattr(parent, attr, factory(old))


def get_in_out_dims(linear_or_conv1d: nn.Module) -> tuple[int, int]:
    """统一返回 (in_features, out_features)，兼容 nn.Linear 和 GPT-2 Conv1D。

    GPT-2 Conv1D.weight shape = (in, out)；nn.Linear.weight shape = (out, in)。
    """
    from transformers.pytorch_utils import Conv1D
    if isinstance(linear_or_conv1d, Conv1D):
        d_in, d_out = linear_or_conv1d.weight.shape
    elif isinstance(linear_or_conv1d, nn.Linear):
        d_out, d_in = linear_or_conv1d.weight.shape
    else:
        raise TypeError(f"Unsupported module type: {type(linear_or_conv1d)}")
    return d_in, d_out


def is_conv1d(m: nn.Module) -> bool:
    from transformers.pytorch_utils import Conv1D
    return isinstance(m, Conv1D)


# ==============================
# Toy 数据集（与 prompt-tuning-family 一致）
# ==============================
class ToySentimentDataset(torch.utils.data.Dataset):
    """简单情感数据集，用于 mini training（forward / loss 验证）。"""
    _POSITIVE = [
        "This movie is amazing and beautiful",
        "I love this product, highly recommended",
        "Wonderful story and great acting",
        "Best book I have read this year",
        "Fantastic experience from start to finish",
    ]
    _NEGATIVE = [
        "Terrible, waste of time and money",
        "I hate this, absolutely awful",
        "Poor quality and bad service",
        "Worst purchase ever, very disappointed",
        "Boring, predictable and badly written",
    ]

    def __init__(self, tokenizer, max_length: int = 32, n_repeat: int = 4, seed: int = 42):
        rng = random.Random(seed)
        texts, labels = [], []
        for _ in range(n_repeat):
            for t in self._POSITIVE:
                texts.append(t)
                labels.append(1)
            for t in self._NEGATIVE:
                texts.append(t)
                labels.append(0)
        idx = list(range(len(texts)))
        rng.shuffle(idx)
        self.texts = [texts[i] for i in idx]
        self.labels = [labels[i] for i in idx]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": enc["input_ids"][0].clone(),  # LM 训练
        }
