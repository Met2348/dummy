"""(IA)³ 最小实现（手写）。

对应论文: Liu et al. 2022, "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning" (NeurIPS)
对应 lecture: lectures/05-ia3.md

核心公式（极致压缩）：
    每个 transformer block 学 3 个对角缩放向量 l_k, l_v, l_ff
    并把它们逐元素乘到 attention 和 FFN 的输出上

    Attention (在 c_attn 后):
        Q, K, V = split(c_attn(x), 3)
        K *= l_k     ← (d,) 向量逐元素乘
        V *= l_v     ← (d,) 向量逐元素乘

    FFN (在 c_fc 后，激活前):
        h = c_fc(x)
        h *= l_ff    ← (intermediate,) 向量逐元素乘
        h = act(h)
        out = c_proj(h)

参数量 per layer:
    d_k + d_v + d_ff = 768 + 768 + 3072 = 4,608

总参数 (GPT-2):
    12 * 4,608 = 55,296

→ 比 LoRA r=8 (294,912) 小 5×，比 Pfeiffer (304K) 小 5.5×
→ 极致压缩，但效果在 few-shot 上接近全参 FT
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent))
from common import freeze_base_model, print_param_summary  # noqa: E402


class _IA3AttnWrapper(nn.Module):
    """把 IA3 的 l_k, l_v 应用到 GPT2Attention 的 c_attn 输出。

    GPT-2 的 c_attn 是 Conv1D(d → 3d)，输出 [Q, K, V] 拼接。
    我们包装 c_attn：在它输出后把后两段 (K, V) 缩放。
    """

    def __init__(self, c_attn: nn.Module, d: int):
        super().__init__()
        for p in c_attn.parameters():
            p.requires_grad = False
        self.c_attn = c_attn
        self.d = d
        self.l_k = nn.Parameter(torch.ones(d))
        self.l_v = nn.Parameter(torch.ones(d))

    def forward(self, x):
        qkv = self.c_attn(x)  # (..., 3d)
        # split into Q, K, V
        q, k, v = qkv.split(self.d, dim=-1)
        k = k * self.l_k
        v = v * self.l_v
        return torch.cat([q, k, v], dim=-1)


class _IA3MlpWrapper(nn.Module):
    """IA3 的 l_ff 应用到 FFN 中间激活。

    GPT-2 MLP: c_fc (d → 4d) → act → c_proj (4d → d)
    """

    def __init__(self, base_mlp: nn.Module):
        super().__init__()
        for p in base_mlp.parameters():
            p.requires_grad = False
        self.base = base_mlp
        # c_fc 输出维度 = 4d = 3072
        intermediate = base_mlp.c_fc.weight.shape[1]  # Conv1D weight shape (in, out)
        self.l_ff = nn.Parameter(torch.ones(intermediate))

    def forward(self, x):
        h = self.base.c_fc(x)
        h = h * self.l_ff  # IA3 缩放
        h = self.base.act(h)
        h = self.base.c_proj(h)
        if hasattr(self.base, "dropout"):
            h = self.base.dropout(h)
        return h


class IA3GPT2(nn.Module):
    """GPT-2 + (IA)³ 缩放向量。"""

    def __init__(self, base_model_name: str = "gpt2"):
        super().__init__()
        self.lm = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        freeze_base_model(self.lm)

        d = self.lm.config.n_embd  # 768

        for block in self.lm.transformer.h:
            # 替换 c_attn 为 IA3 包装版
            block.attn.c_attn = _IA3AttnWrapper(block.attn.c_attn, d)
            # 替换 mlp 为 IA3 包装版
            block.mlp = _IA3MlpWrapper(block.mlp)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )


def main() -> None:
    torch.manual_seed(42)
    model = IA3GPT2()
    print_param_summary(model, "(IA)^3")
    # Expected per layer: 768 + 768 + 3072 = 4,608
    # 12 layers: 55,296
    print(f"\n参数布局 per layer:")
    print(f"  l_k:  768")
    print(f"  l_v:  768")
    print(f"  l_ff: 3072")
    print(f"  per layer: 4,608")
    print(f"  12 layer 合计: 55,296")

    # 验证初始 forward = base（因为所有缩放向量初始为 1）
    base = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    model.eval()
    enc = model.tokenizer("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out_i = model(enc["input_ids"], enc["attention_mask"])
        out_b = base(enc["input_ids"], attention_mask=enc["attention_mask"])
    diff = (out_i.logits - out_b.logits).abs().max().item()
    print(f"\n初始 forward vs base 误差: {diff:.4e}")
    print("  → 所有缩放向量初始为 1 (identity)")


if __name__ == "__main__":
    main()
