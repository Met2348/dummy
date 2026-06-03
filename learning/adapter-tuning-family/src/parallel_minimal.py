"""Parallel Adapter 最小实现（手写）。

对应论文: He et al. 2022, "Towards a Unified View of Parameter-Efficient Transfer Learning" (ICLR)
对应 lecture: lectures/04-parallel-adapter.md

核心差异（串联 vs 并联）：
    串联 (Houlsby/Pfeiffer):
        x → base(x) → adapter(base(x)) → out

    并联 (Parallel Adapter):
        x → base(x)
        x → adapter(x)
        out = base(x) + adapter(x) * scaling

公式:
    out = base(x) + s * (W_up * σ(W_down * x))

→ 这与 LoRA 的差异仅在于"是否有非线性 σ"
   去掉 σ → 退化为 LoRA
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent))
from common import freeze_base_model, print_param_summary  # noqa: E402


class ParallelAdapter(nn.Module):
    """单个 Parallel Adapter: out = base(x) + s * up(σ(down(x)))

    参数:
        d_in, d_out: 输入/输出维度
        r: bottleneck
        scaling: 缩放因子（类似 LoRA 的 α/r）
    """

    def __init__(self, base_module: nn.Module, r: int = 16, scaling: float = 1.0):
        super().__init__()
        # 冻结 base
        for p in base_module.parameters():
            p.requires_grad = False
        self.base = base_module

        # 推断维度
        if hasattr(base_module, "in_features"):
            d_in, d_out = base_module.in_features, base_module.out_features
        elif base_module.__class__.__name__ == "Conv1D":
            d_in, d_out = base_module.weight.shape  # Conv1D: (in, out)
        elif base_module.__class__.__name__ == "GPT2MLP":
            # GPT-2 的 MLP 是 d→d 映射
            d_in = d_out = base_module.c_fc.weight.shape[0]  # Conv1D (in, out=4d)
        else:
            raise ValueError(f"Unknown base type: {base_module.__class__.__name__}")

        self.down = nn.Linear(d_in, r)
        self.up = nn.Linear(r, d_out)
        self.act = nn.GELU()
        self.scaling = scaling

        # 零初始化保证初始 forward = base
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        adapter_out = self.up(self.act(self.down(x))) * self.scaling
        return base_out + adapter_out


class ParallelAdapterGPT2(nn.Module):
    """GPT-2 + Parallel Adapter（应用在 FFN 上）。

    与 Pfeiffer 区别：
        Pfeiffer: 在 FFN 后串联一个 adapter
        Parallel: 在 FFN 并联一个 adapter (与 FFN 共享输入 x，输出相加)
    """

    def __init__(
        self,
        base_model_name: str = "gpt2",
        r: int = 16,
        scaling: float = 1.0,
    ):
        super().__init__()
        self.lm = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        freeze_base_model(self.lm)

        self.r = r
        self.scaling = scaling

        # 用 ParallelAdapter 替换每个 block 的 MLP
        for block in self.lm.transformer.h:
            block.mlp = ParallelAdapter(block.mlp, r=r, scaling=scaling)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )


def main() -> None:
    torch.manual_seed(42)
    model = ParallelAdapterGPT2(r=16, scaling=1.0)
    print_param_summary(model, "Parallel Adapter (r=16)")
    # GPT-2 MLP 的 in/out 都是 768，所以 down/up 形状与 Pfeiffer 一样
    # 参数量 = Pfeiffer
    print(f"\n参数布局：与 Pfeiffer 完全相同")
    print(f"  per layer: 2 * (768*16 + bias) = 25,360")
    print(f"  12 layer: 304,320")

    # 验证初始 forward = base
    from transformers import GPT2LMHeadModel
    base = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    model.eval()
    enc = model.tokenizer("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out_p = model(enc["input_ids"], enc["attention_mask"])
        out_b = base(enc["input_ids"], attention_mask=enc["attention_mask"])
    diff = (out_p.logits - out_b.logits).abs().max().item()
    print(f"\n初始 forward vs base 误差: {diff:.4e}")
    print("  → up 层零初始化保证 adapter 输出 = 0")


if __name__ == "__main__":
    main()
