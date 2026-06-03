"""Pfeiffer Adapter 最小实现（手写）。

对应论文: Pfeiffer et al. 2020, "AdapterFusion: Non-Destructive Task Composition for Transfer Learning"
对应 lecture: lectures/01-houlsby-pfeiffer.md

核心简化：相比 Houlsby（每 block 插 2 个 adapter），Pfeiffer 只插 1 个（FFN 后）。
    参数量 ≈ Houlsby 的 1/2
    实验上效果相当（Pfeiffer 论文指出"only second adapter matters"）

公式同 Houlsby:
    Adapter(x) = up(act(down(x))) + x
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


class PfeifferGPT2(nn.Module):
    """GPT-2 + Pfeiffer Adapter（每 block 只插 1 个，在 FFN 后）。"""

    def __init__(
        self,
        base_model_name: str = "gpt2",
        r: int = 16,
    ):
        super().__init__()
        self.lm = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        freeze_base_model(self.lm)

        d = self.lm.config.n_embd
        self.r = r

        # 只在 FFN 后插 adapter（Pfeiffer 简化）
        for block in self.lm.transformer.h:
            adapter_ffn = HoulsbyAdapter(d, r)  # 复用 HoulsbyAdapter 结构
            block.mlp = _MlpAdapterWrapper(block.mlp, adapter_ffn)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )


def main() -> None:
    torch.manual_seed(42)
    model = PfeifferGPT2(r=16)
    print_param_summary(model, "Pfeiffer Adapter (r=16)")
    # Expected: 半个 Houlsby = 12 × 25,360 = 304,320
    print(f"\n参数布局: 12 × (down + up) = 12 × 25,360 = 304,320")
    print(f"  约 Houlsby 一半（少了 attn 后的 adapter）")

    # 验证初始 forward = base
    base = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    model.eval()
    enc = model.tokenizer("hello world this is a test", return_tensors="pt", padding=True)
    with torch.no_grad():
        out_p = model(enc["input_ids"], enc["attention_mask"])
        out_b = base(enc["input_ids"], attention_mask=enc["attention_mask"])
    diff = (out_p.logits - out_b.logits).abs().max().item()
    print(f"\n初始 forward 与 base 最大误差: {diff:.4e}")


if __name__ == "__main__":
    main()
