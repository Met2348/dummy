"""Adapter Tuning Family 通用工具函数。

复用 LoRA 系列的 freeze/get_in_out_dims/target_linear_modules，并新增 adapter 专用 helper。
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ===== 基础工具（与 LoRA 系列保持一致）=====

def is_conv1d(module: nn.Module) -> bool:
    """检测 transformers 的 Conv1D（GPT-2 用）。"""
    return module.__class__.__name__ == "Conv1D"


def get_in_out_dims(linear: nn.Module) -> tuple[int, int]:
    """统一返回 (d_in, d_out)，无论 nn.Linear 还是 Conv1D。

    - nn.Linear.weight shape = (d_out, d_in)
    - Conv1D.weight shape = (d_in, d_out)
    """
    if is_conv1d(linear):
        d_in, d_out = linear.weight.shape
    else:
        d_out, d_in = linear.weight.shape
    return d_in, d_out


def freeze_base_model(model: nn.Module) -> None:
    """冻结所有参数（adapter 训练前置步骤）。"""
    for p in model.parameters():
        p.requires_grad = False


def target_linear_modules(
    model: nn.Module, target_names: tuple[str, ...]
) -> list[tuple[str, nn.Module]]:
    """遍历找到所有名称包含 target_names 的 Linear/Conv1D 模块。

    返回 [(qualified_name, module), ...]
    """
    matches = []
    for qname, mod in model.named_modules():
        if not (isinstance(mod, nn.Linear) or is_conv1d(mod)):
            continue
        if any(t in qname for t in target_names):
            matches.append((qname, mod))
    return matches


def get_parent_and_attr(model: nn.Module, qname: str) -> tuple[nn.Module, str]:
    """'transformer.h.0.attn.c_attn' → (h_0_attn_module, 'c_attn')。"""
    parts = qname.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def print_param_summary(model: nn.Module, name: str = "model") -> None:
    """打印模型参数量（总/可训练/冻结）。"""
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - train
    pct = 100 * train / total if total > 0 else 0
    print(f"\n  {name}")
    print(f"  Total params:      {total:>12,}")
    print(f"  Trainable params:  {train:>12,}  ({pct:.4f}%)")
    print(f"  Frozen params:     {frozen:>12,}")


# ===== Adapter 专用 helper =====

class AdapterHook(nn.Module):
    """让 Adapter 模块"挂"在某个 base module 后面。

    通过 register_forward_hook 实现，避免直接替换 base module。
    用于 Houlsby / Pfeiffer 等 post-module adapter 插入。
    """

    def __init__(self, adapter: nn.Module):
        super().__init__()
        self.adapter = adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)


def attach_adapter_after(base_module: nn.Module, adapter: nn.Module) -> nn.Module:
    """通过 sequential 把 adapter 接在 base_module 后面。"""
    return nn.Sequential(base_module, adapter)


# ===== Toy 数据集（mini training 通用） =====

class ToySentimentDataset:
    """4 样本极简数据集，仅用于演示 mini training loss 下降。"""

    SAMPLES = [
        ("hello world this is a test", 0),
        ("i love this amazing product", 1),
        ("absolutely terrible movie", 0),
        ("fantastic experience from start to finish", 1),
    ]

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_batch(self):
        texts = [s[0] for s in self.SAMPLES]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True)
        enc["labels"] = enc["input_ids"].clone()
        return enc
