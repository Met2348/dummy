"""
P-Tuning v2 最小实现（手写）。

对应论文：Liu et al., 2022, arXiv:2110.07602
对应 lecture: lectures/04-p-tuning-v2.md

核心思想：
  - 冻结 GPT-2
  - 每层 KV 前加可训练 prefix（思想同 Prefix Tuning）
  - 关键差异：**不用 MLP reparameterization，直接学**
  - 简化版：本实现不加分类头，仅演示算法骨架

与 prefix_tuning_minimal.py 的差异：
  - 无 reparam（self.reparam 替换为直接学的 self.prefix）
  - prefix 张量形状 (L, 2, p, n_head, d_h)

公式索引：
    P^(l, K), P^(l, V) ∈ R^(p, d)              (1, 2)
    P 直接学（init from N(0, 0.02^2)）          (3)
    ŷ = softmax(W_cls h_cls)                   (4)   [本 minimal 简化省略]
    head_h^(l) = Attn(Q, [P^K; K], [P^V; V])   (5)
    φ* = argmin CE(ŷ, y)                       (6)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import DynamicCache, GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent))
from common import print_param_summary  # noqa: E402


class PTuningV2GPT2(nn.Module):
    """P-Tuning v2 包装器。

    与 Prefix Tuning 的核心差异：无 MLP reparameterization，直接学 prefix。
    """

    def __init__(
        self,
        base_model_name: str = "gpt2",
        prefix_length: int = 10,
    ):
        super().__init__()
        self.lm = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 公式 (6): θ_LM 冻结
        for p in self.lm.parameters():
            p.requires_grad = False

        cfg = self.lm.config
        self.n_layer = cfg.n_layer       # L
        self.n_head = cfg.n_head         # H
        self.embed_dim = cfg.n_embd      # d
        self.head_dim = self.embed_dim // self.n_head   # d_h
        self.prefix_length = prefix_length

        # 公式 (1)(2): 直接学 (L, 2, p, n_head, head_dim) 大张量
        # 不用 MLP，与 Prefix Tuning 的关键差异
        self.prefix = nn.Parameter(
            torch.empty(
                self.n_layer, 2, prefix_length, self.n_head, self.head_dim
            )
        )
        nn.init.normal_(self.prefix, mean=0.0, std=0.02)

    def get_past_key_values(self, batch_size: int) -> DynamicCache:
        """生成 DynamicCache。

        prefix shape: (L, 2, p, n_head, d_h)
        每层输出 (K, V): (B, n_head, p, d_h)
        """
        past = []
        for layer_idx in range(self.n_layer):
            # (p, n_head, d_h) → (n_head, p, d_h) → (B, n_head, p, d_h)
            k = self.prefix[layer_idx, 0].permute(1, 0, 2)
            v = self.prefix[layer_idx, 1].permute(1, 0, 2)
            k = k.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
            v = v.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
            past.append((k, v))
        return DynamicCache(ddp_cache_data=past)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: torch.LongTensor | None = None,
    ):
        """前向：公式 (5) → (6)。"""
        B = input_ids.shape[0]
        past = self.get_past_key_values(B)

        prefix_mask = torch.ones(
            B, self.prefix_length, dtype=attention_mask.dtype, device=attention_mask.device
        )
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        return self.lm(
            input_ids=input_ids,
            attention_mask=full_mask,
            past_key_values=past,
            labels=labels,
            use_cache=False,
        )


def main() -> None:
    torch.manual_seed(42)
    model = PTuningV2GPT2(prefix_length=10)
    print_param_summary(model, "PTuningV2GPT2(p=10, 无 reparam)")
    # Expected: trainable = L*p*2*d = 12*10*2*768 = 184,320

    print(f"\n参数布局：")
    print(f"  prefix shape: {tuple(model.prefix.shape)}")
    print(f"  numel = L * 2 * p * H * d_h = {model.n_layer}*2*{model.prefix_length}*"
          f"{model.n_head}*{model.head_dim} = {model.prefix.numel():,}")

    enc = model.tokenizer("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"])
    print(f"\n前向输出 logits.shape={tuple(out.logits.shape)}")


if __name__ == "__main__":
    main()
