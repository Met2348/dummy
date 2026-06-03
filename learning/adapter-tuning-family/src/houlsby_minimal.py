"""Houlsby Adapter 最小实现（手写）。

对应论文: Houlsby et al. 2019, "Parameter-Efficient Transfer Learning for NLP"
对应 lecture: lectures/01-houlsby-pfeiffer.md

核心：
    Adapter(x) = up(act(down(x))) + x   ← 残差
        down: d → r （bottleneck 降维）
        up:   r → d （还原）
        act:  GELU/ReLU 等非线性

Houlsby 双串联：每个 transformer block 插 2 个 adapter
    - 一个在 Attention 之后
    - 一个在 FFN 之后

参数量 per layer:
    2 × (d × r + r × d + biases) ≈ 4 × d × r
    GPT-2 d=768, r=16: 2 × (768*16 + 16*768) ≈ 49K per layer
    12 层共约 600K（base 124M 的 0.5%）
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent))
from common import freeze_base_model, print_param_summary  # noqa: E402


class HoulsbyAdapter(nn.Module):
    """单个 Houlsby Adapter: down → act → up → +residual

    参数:
        d: 输入/输出维度（同 base 模型）
        r: bottleneck 维度（远小于 d）
        act: 非线性激活，默认 GELU
    """

    def __init__(self, d: int, r: int = 16, act: str = "gelu"):
        super().__init__()
        self.down = nn.Linear(d, r)
        self.up = nn.Linear(r, d)
        self.act = nn.GELU() if act == "gelu" else nn.ReLU()

        # 关键：up 层零初始化 → 初始 forward = base + 0 = base
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down(x)
        h = self.act(h)
        h = self.up(h)
        return x + h  # 残差连接


class HoulsbyBlock(nn.Module):
    """包装 GPT2Block：在 attention 输出后 + FFN 输出后各插一个 adapter。

    GPT2Block forward:
        h1 = ln_1(x); h2 = attn(h1); x = x + h2          ← 在这里插 adapter_attn
        h3 = ln_2(x); h4 = mlp(h3); x = x + h4           ← 在这里插 adapter_ffn
    """

    def __init__(self, base_block: nn.Module, d: int, r: int = 16):
        super().__init__()
        self.base_block = base_block
        self.adapter_attn = HoulsbyAdapter(d, r)
        self.adapter_ffn = HoulsbyAdapter(d, r)

    def forward(self, hidden_states, *args, **kwargs):
        # 复制 GPT2Block.forward 逻辑，但在两处插入 adapter
        # 注意：transformers GPT2Block 的 attention 和 mlp 调用需要适配 layer_past, attention_mask 等
        # 这里用 wrapper 的方式：调用原始 block，然后在 attention 和 mlp 输出后挂 adapter
        # 简化处理：让 base_block 自己处理，但通过 hook 注入 adapter
        # 但这种方式难以"在两个残差之间插入"，所以我们直接 monkey-patch base_block 的 attn/mlp 输出
        return self.base_block(hidden_states, *args, **kwargs)


class _AttnAdapterWrapper(nn.Module):
    """包装 GPT2Attention：在原 forward 输出后挂 adapter。"""

    def __init__(self, base_attn: nn.Module, adapter: nn.Module):
        super().__init__()
        self.base_attn = base_attn
        self.adapter = adapter

    def forward(self, *args, **kwargs):
        outputs = self.base_attn(*args, **kwargs)
        # transformers GPT2Attention 返回 (attn_output, present, attn_weights)
        # 我们需要在 attn_output 上应用 adapter
        attn_output = outputs[0]
        attn_output = self.adapter(attn_output)
        return (attn_output,) + outputs[1:]


class _MlpAdapterWrapper(nn.Module):
    """包装 GPT2MLP：在原 forward 输出后挂 adapter。"""

    def __init__(self, base_mlp: nn.Module, adapter: nn.Module):
        super().__init__()
        self.base_mlp = base_mlp
        self.adapter = adapter

    def forward(self, x):
        h = self.base_mlp(x)
        h = self.adapter(h)
        return h


class HoulsbyGPT2(nn.Module):
    """GPT-2 + Houlsby Adapter（每 block 插 2 个）。"""

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

        d = self.lm.config.n_embd  # 768 for gpt2
        self.r = r

        # 在每个 transformer block 插入 2 个 adapter
        for block in self.lm.transformer.h:
            adapter_attn = HoulsbyAdapter(d, r)
            adapter_ffn = HoulsbyAdapter(d, r)
            block.attn = _AttnAdapterWrapper(block.attn, adapter_attn)
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
    model = HoulsbyGPT2(r=16)
    print_param_summary(model, "Houlsby Adapter (r=16)")
    # Expected:
    #   每层 2 个 adapter，每个 adapter: 2 × (768×16 + 16) + (16×768 + 768) ≈ 25K
    #   12 层 × 2 ≈ 600K
    # 实际：
    #   adapter_attn: down (768→16) = 12,304 + up (16→768) = 13,056 = 25,360
    #   adapter_ffn: 同上 = 25,360
    #   12 × 2 × 25,360 = 608,640
    print(f"\n参数布局 per adapter:")
    print(f"  down (768→16): 768*16 + 16 = 12,304")
    print(f"  up   (16→768): 16*768 + 768 = 13,056")
    print(f"  total: 25,360")
    print(f"  每 block 2 个 adapter = 50,720")
    print(f"  12 block 合计 = 608,640")

    # 验证初始 forward = base
    base = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    model.eval()
    enc = model.tokenizer("hello world this is a test", return_tensors="pt", padding=True)
    with torch.no_grad():
        out_h = model(enc["input_ids"], enc["attention_mask"])
        out_b = base(enc["input_ids"], attention_mask=enc["attention_mask"])
    diff = (out_h.logits - out_b.logits).abs().max().item()
    print(f"\n初始 forward 与 base 最大误差: {diff:.4e}")
    print("  → up 层零初始化保证 Adapter(x) = x + 0 = x")


if __name__ == "__main__":
    main()
