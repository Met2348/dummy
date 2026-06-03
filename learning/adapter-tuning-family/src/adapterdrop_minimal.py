"""AdapterDrop 最小实现（手写）。

对应论文: Rücklé et al. 2020, "AdapterDrop: On the Efficiency of Adapters" (EMNLP)
对应 lecture: lectures/03-adapterdrop-compacter.md

核心思想：
    训练时随机丢弃浅层 adapter，推理时丢前 k 层
    → 推理加速 10-30%（lower layers 算 adapter 但效果有限）

策略:
    1. 训练阶段: P(drop) 概率丢 adapter（一层或多层）
    2. 推理阶段: 永久丢前 k 层（k=5 时 ~25% 加速）
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent))
from common import freeze_base_model, print_param_summary  # noqa: E402
from houlsby_minimal import HoulsbyAdapter, _MlpAdapterWrapper  # noqa: E402


class _DroppableMlpWrapper(nn.Module):
    """可丢弃的 Adapter wrapper。

    支持两种 drop 策略:
        - training: 用 self.drop_prob 概率丢
        - inference: 用 self.permanent_drop bool 控制
    """

    def __init__(self, base_mlp, adapter, layer_idx: int):
        super().__init__()
        self.base_mlp = base_mlp
        self.adapter = adapter
        self.layer_idx = layer_idx
        self.drop_prob = 0.0  # 训练时随机丢概率
        self.permanent_drop = False  # 推理时永久丢

    def forward(self, x):
        h = self.base_mlp(x)
        if self.permanent_drop:
            return h
        if self.training and self.drop_prob > 0 and torch.rand(1).item() < self.drop_prob:
            return h
        return self.adapter(h)


class AdapterDropGPT2(nn.Module):
    """GPT-2 + Pfeiffer Adapter + AdapterDrop 机制。"""

    def __init__(
        self,
        base_model_name: str = "gpt2",
        r: int = 16,
        train_drop_prob: float = 0.1,
    ):
        super().__init__()
        self.lm = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        freeze_base_model(self.lm)

        d = self.lm.config.n_embd
        self.r = r
        self.n_layers = len(self.lm.transformer.h)
        self.train_drop_prob = train_drop_prob

        # 每个 block 挂一个 droppable adapter
        for i, block in enumerate(self.lm.transformer.h):
            adapter = HoulsbyAdapter(d, r)
            wrapper = _DroppableMlpWrapper(block.mlp, adapter, i)
            wrapper.drop_prob = train_drop_prob
            block.mlp = wrapper

    def set_inference_drop(self, k: int) -> None:
        """推理时永久丢前 k 层 adapter。"""
        for i, block in enumerate(self.lm.transformer.h):
            block.mlp.permanent_drop = (i < k)

    def get_active_layers(self) -> int:
        """统计当前激活 adapter 层数。"""
        return sum(1 for b in self.lm.transformer.h if not b.mlp.permanent_drop)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )


def main() -> None:
    torch.manual_seed(42)
    model = AdapterDropGPT2(r=16, train_drop_prob=0.1)
    print_param_summary(model, "AdapterDrop (r=16, drop_p=0.1)")

    print(f"\n训练时: 每个 adapter 有 10% 概率被丢")
    print(f"推理时: 可手动丢前 k 层")
    print(f"  k=0:  {model.get_active_layers()} 层 active")
    model.set_inference_drop(k=5)
    print(f"  k=5:  {model.get_active_layers()} 层 active (前 5 层 adapter 不算)")
    model.set_inference_drop(k=11)
    print(f"  k=11: {model.get_active_layers()} 层 active")


if __name__ == "__main__":
    main()
