"""AdapterFusion 最小实现（手写）。

对应论文: Pfeiffer et al. 2021, "AdapterFusion: Non-Destructive Task Composition" (EACL)
对应 lecture: lectures/02-adapterfusion.md

核心思想（两阶段）：
    Stage 1: 单独训每个 task adapter（冻 base）
    Stage 2: 冻所有 task adapter，加 attention 融合层学习如何组合它们

Fusion 公式（每 block 一个 fusion 层）：
    输入: x (hidden), {a_1(x), ..., a_N(x)} (N 个冻结 adapter 的输出)
    Q = W_q * x                    ← 用 hidden 做 query
    K = [W_k * a_1, ..., W_k * a_N]
    V = [W_v * a_1, ..., W_v * a_N]
    α = softmax(Q · K^T / √d)
    fused = Σ α_i * V_i

→ 模型动态学习"在当前 hidden 下，应该多用哪个 adapter"。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent))
from common import freeze_base_model, print_param_summary  # noqa: E402
from houlsby_minimal import HoulsbyAdapter, _MlpAdapterWrapper  # noqa: E402


class FusionLayer(nn.Module):
    """单个 Fusion 层：用 attention 融合 N 个 frozen adapter 的输出。

    参数量: 3 × d² (Q, K, V)，与 adapter 大小无关。
    """

    def __init__(self, d: int, n_adapters: int):
        super().__init__()
        self.d = d
        self.n_adapters = n_adapters
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        # value 投影通常用 identity 初始化（避免初始破坏）
        nn.init.eye_(self.W_v.weight)
        # query/key 用小的随机值
        nn.init.normal_(self.W_q.weight, std=0.02)
        nn.init.normal_(self.W_k.weight, std=0.02)

    def forward(self, x: torch.Tensor, adapter_outs: list[torch.Tensor]) -> torch.Tensor:
        """
        x: (batch, seq, d) — 当前 hidden
        adapter_outs: list of N tensors，每个 (batch, seq, d)
        """
        Q = self.W_q(x)  # (b, s, d)
        # stack adapter outputs: (b, s, N, d)
        A = torch.stack(adapter_outs, dim=2)
        K = self.W_k(A)  # (b, s, N, d)
        V = self.W_v(A)  # (b, s, N, d)
        # attention scores: (b, s, N) = Q (b,s,d) · K (b,s,N,d)
        scores = (Q.unsqueeze(2) * K).sum(dim=-1) / math.sqrt(self.d)
        attn = F.softmax(scores, dim=-1)  # (b, s, N)
        # weighted sum: Σ attn_i * V_i
        fused = (attn.unsqueeze(-1) * V).sum(dim=2)  # (b, s, d)
        return fused


class _FusionMlpWrapper(nn.Module):
    """在 MLP 输出后，用 fusion 融合多个 frozen adapter 的输出。"""

    def __init__(self, base_mlp: nn.Module, adapters: nn.ModuleList, fusion: FusionLayer):
        super().__init__()
        self.base_mlp = base_mlp
        self.adapters = adapters  # 冻结的 N 个 adapter
        self.fusion = fusion  # 唯一可训练

    def forward(self, x):
        h = self.base_mlp(x)
        adapter_outs = [a(h) for a in self.adapters]  # N 个输出
        fused = self.fusion(h, adapter_outs)
        return fused


class AdapterFusionGPT2(nn.Module):
    """GPT-2 + N 个 frozen task adapter + Fusion 层。

    使用流程:
        1. 先单独训 N 个 Pfeiffer adapter (用 PfeifferGPT2)
        2. 加载它们的权重，冻结
        3. 加 Fusion 层，只训这个
    """

    def __init__(
        self,
        base_model_name: str = "gpt2",
        n_adapters: int = 3,
        r: int = 16,
    ):
        super().__init__()
        self.lm = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        freeze_base_model(self.lm)

        d = self.lm.config.n_embd
        self.n_adapters = n_adapters
        self.r = r

        # 为每个 block 创建 N 个 adapter + 1 个 fusion 层
        for block in self.lm.transformer.h:
            adapters = nn.ModuleList([HoulsbyAdapter(d, r) for _ in range(n_adapters)])
            # 冻 adapters（模拟"预训练 adapter"）
            for a in adapters:
                for p in a.parameters():
                    p.requires_grad = False
            fusion = FusionLayer(d, n_adapters)
            block.mlp = _FusionMlpWrapper(block.mlp, adapters, fusion)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )


def main() -> None:
    torch.manual_seed(42)
    model = AdapterFusionGPT2(n_adapters=3, r=16)
    print_param_summary(model, "AdapterFusion (N=3, r=16)")
    # 只有 fusion 可训练: 12 层 × 3 × 768² = ~21M
    print(f"\n参数布局:")
    print(f"  每层 fusion: 3 * 768 * 768 = 1,769,472")
    print(f"  12 层合计: 21,233,664")
    print(f"  注意: adapters 冻结（已预训好），只训 fusion")

    enc = model.tokenizer("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"])
    print(f"\nforward logits.shape={tuple(out.logits.shape)}")


if __name__ == "__main__":
    main()
