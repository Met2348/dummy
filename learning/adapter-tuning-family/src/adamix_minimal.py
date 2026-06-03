"""AdaMix 最小实现（手写）。

对应论文: Wang et al. 2022, "AdaMix: Mixture-of-Adaptations for Parameter-efficient Model Tuning" (EMNLP)
对应 lecture: lectures/08-adamix.md

核心思想 — Mixture of Adapters (MoA):
    每个 transformer block 有 N 个 adapter expert
    训练时: 随机选 1 个 expert 处理这个 token/batch (stochastic routing)
    推理时: 取所有 expert 输出的平均 (weight averaging)

特点:
    - 类 MoE 但更轻（adapter 而非 layer）
    - 训练高效（每次只算 1 个 expert）
    - 推理时 average 减少 variance

参数 per layer: N × (adapter_params)
    GPT-2 d=768, r=16, N=4: 4 × 25,360 = 101,440
    12 layer: 1,217,280
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


class AdaMixLayer(nn.Module):
    """Mixture of N adapter experts。

    训练: 每个 forward 随机选一个 expert
    推理: 所有 expert 输出取平均
    """

    def __init__(self, d: int, r: int = 16, n_experts: int = 4):
        super().__init__()
        self.experts = nn.ModuleList([
            HoulsbyAdapter(d, r) for _ in range(n_experts)
        ])
        self.n_experts = n_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Stochastic routing: 随机选 1 个 expert
            idx = torch.randint(0, self.n_experts, (1,)).item()
            return self.experts[idx](x)
        else:
            # Inference: weight averaging
            outs = [e(x) for e in self.experts]
            return torch.stack(outs).mean(dim=0)


class _AdaMixMlpWrapper(nn.Module):
    """挂载 AdaMix（替代 Pfeiffer 的单 adapter）。"""

    def __init__(self, base_mlp, adamix_layer):
        super().__init__()
        self.base_mlp = base_mlp
        self.adamix = adamix_layer

    def forward(self, x):
        h = self.base_mlp(x)
        return self.adamix(h)


class AdaMixGPT2(nn.Module):
    """GPT-2 + AdaMix（每 block 一个 Mixture-of-Adapters）。"""

    def __init__(
        self,
        base_model_name: str = "gpt2",
        r: int = 16,
        n_experts: int = 4,
    ):
        super().__init__()
        self.lm = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        freeze_base_model(self.lm)

        d = self.lm.config.n_embd
        self.r = r
        self.n_experts = n_experts

        for block in self.lm.transformer.h:
            adamix = AdaMixLayer(d, r, n_experts)
            block.mlp = _AdaMixMlpWrapper(block.mlp, adamix)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def merge_experts(self) -> None:
        """推理优化：把所有 expert 平均成单个 expert（论文最终步骤）。"""
        for block in self.lm.transformer.h:
            adamix = block.mlp.adamix
            # 平均 down 和 up
            for attr in ("down", "up"):
                weights = torch.stack([
                    getattr(e, attr).weight for e in adamix.experts
                ]).mean(dim=0)
                biases = torch.stack([
                    getattr(e, attr).bias for e in adamix.experts
                ]).mean(dim=0)
                getattr(adamix.experts[0], attr).weight.data.copy_(weights)
                getattr(adamix.experts[0], attr).bias.data.copy_(biases)
            # 只留第 0 个 expert
            adamix.experts = nn.ModuleList([adamix.experts[0]])
            adamix.n_experts = 1


def main() -> None:
    torch.manual_seed(42)
    model = AdaMixGPT2(r=16, n_experts=4)
    print_param_summary(model, "AdaMix (4 experts, r=16)")
    # Expected: 4 × 12 × 25,360 = 1,217,280
    print(f"\n参数布局: 4 experts × 12 layer × 25,360 = 1,217,280")

    # 验证训练时随机路由
    model.train()
    enc = model.tokenizer("hello world", return_tensors="pt", padding=True)
    enc["labels"] = enc["input_ids"].clone()

    print(f"\n训练时随机路由测试（5 次相同输入）:")
    losses = []
    torch.manual_seed(123)
    for i in range(5):
        out = model(**enc)
        losses.append(out.loss.item())
    print(f"  losses: {losses}")
    print("  → 不同步骤随机选不同 expert，loss 有差异")

    # 推理时 averaging
    model.eval()
    with torch.no_grad():
        out1 = model(**enc).loss.item()
        out2 = model(**enc).loss.item()
    print(f"\n推理时（averaging）:")
    print(f"  loss: {out1:.4f} == {out2:.4f}")
    print("  → 推理 deterministic")

    # merge experts
    print("\nmerge_experts:")
    model.merge_experts()
    after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  合并后参数: {after:,} (≈ Pfeiffer 304K)")


if __name__ == "__main__":
    main()
