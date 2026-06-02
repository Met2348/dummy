"""
Prefix Tuning 最小实现（手写，不依赖 peft）。

对应论文：Li & Liang, 2021, arXiv:2101.00190
对应 lecture: lectures/01-prefix-tuning.md

核心思想：
  - 冻结 GPT-2
  - 在每层 self-attention 的 K、V 前拼接可训练 prefix
  - 用 MLP reparameterization 解决直接训不稳的问题
  - transformers 5.x: 用 DynamicCache 注入 past_key_values

公式索引（与 lecture 对齐）：
    Attn(Q, K, V) = softmax(QK^T / sqrt(d_h)) V                    (1)
    head_h^(l) = Attn(Q, [P_K; K], [P_V; V])                       (2)
    [P^(1,K); P^(1,V); ...; P^(L,K); P^(L,V)] = MLP_φ(P'_θ')       (3)
    φ* = argmin -Σ log P(y_t | y_<t, x)                            (4)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import DynamicCache, GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent))
from common import print_param_summary  # noqa: E402


class PrefixTuningGPT2(nn.Module):
    """Prefix Tuning 包装器。

    Args:
        base_model_name: HF 模型名
        prefix_length:   p，prefix 长度（典型 5-20）
        mid_dim:         reparameterization MLP 中间层维度
        use_reparam:     True=有 MLP（论文标准），False=直接学（用于消融）
    """

    def __init__(
        self,
        base_model_name: str = "gpt2",
        prefix_length: int = 10,
        mid_dim: int = 512,
        use_reparam: bool = True,
    ):
        super().__init__()
        self.lm = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 公式 (4): θ_LM 全冻结
        for p in self.lm.parameters():
            p.requires_grad = False

        cfg = self.lm.config
        self.n_layer = cfg.n_layer       # L = 12
        self.n_head = cfg.n_head         # H = 12
        self.embed_dim = cfg.n_embd      # d = 768
        self.head_dim = self.embed_dim // self.n_head   # d_h = 64
        self.prefix_length = prefix_length              # p
        self.use_reparam = use_reparam

        # 公式 (3) 中的 P' ∈ R^(p, d)
        self.P_low = nn.Parameter(torch.empty(prefix_length, self.embed_dim))
        nn.init.normal_(self.P_low, mean=0.0, std=0.02)

        # 公式 (3) 中的 MLP_φ: d → mid → L*2*d
        if use_reparam:
            self.reparam = nn.Sequential(
                nn.Linear(self.embed_dim, mid_dim),
                nn.Tanh(),
                nn.Linear(mid_dim, self.n_layer * 2 * self.embed_dim),
            )
        else:
            # 消融模式：要求 P_low 直接是 (p, L*2*d) 形状
            self.P_low = nn.Parameter(
                torch.empty(prefix_length, self.n_layer * 2 * self.embed_dim)
            )
            nn.init.normal_(self.P_low, mean=0.0, std=0.02)
            self.reparam = nn.Identity()

    def get_past_key_values(self, batch_size: int) -> DynamicCache:
        """生成 transformers 5.x 期望的 DynamicCache。

        每层一个 (K, V) tuple，K、V 形状: (B, n_head, p, head_dim)。
        """
        if self.use_reparam:
            proj = self.reparam(self.P_low)             # (p, L*2*d)
        else:
            proj = self.P_low                            # (p, L*2*d)

        # (p, L*2*d) → (p, L, 2, n_head, head_dim)
        proj = proj.view(
            self.prefix_length, self.n_layer, 2, self.n_head, self.head_dim
        )
        # → (L, 2, n_head, p, head_dim)
        proj = proj.permute(1, 2, 3, 0, 4).contiguous()
        # expand batch: → (L, 2, B, n_head, p, head_dim)
        proj = proj.unsqueeze(2).expand(-1, -1, batch_size, -1, -1, -1)

        # 转成 DynamicCache 期望的 list of (K, V)
        kv_list = []
        for layer_idx in range(self.n_layer):
            k = proj[layer_idx, 0]  # (B, n_head, p, head_dim)
            v = proj[layer_idx, 1]
            kv_list.append((k, v))

        return DynamicCache(ddp_cache_data=kv_list)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: torch.LongTensor | None = None,
    ):
        """前向：公式 (2) → (4)。

        Args:
            input_ids:      (B, n)
            attention_mask: (B, n)
            labels:         (B, n) 或 None

        Returns:
            CausalLMOutputWithCrossAttentions，logits 形状 (B, n, V)。
            注意：与 Prompt Tuning 不同，这里 logits 长度是 n 而非 p+n。
        """
        B = input_ids.shape[0]
        past = self.get_past_key_values(B)

        # 扩展 attention mask：前 p 位为可见
        prefix_mask = torch.ones(
            B, self.prefix_length, dtype=attention_mask.dtype, device=attention_mask.device
        )
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # (B, p+n)

        return self.lm(
            input_ids=input_ids,
            attention_mask=full_mask,
            past_key_values=past,
            labels=labels,
            use_cache=False,
        )


def main() -> None:
    torch.manual_seed(42)
    model = PrefixTuningGPT2(prefix_length=10, mid_dim=512)
    print_param_summary(model, "PrefixTuningGPT2(p=10, mid=512)")
    # Expected: trainable ≈ 9.8M（主要来自 MLP）

    enc = model.tokenizer("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"])
    print(f"\n前向输出 logits.shape={tuple(out.logits.shape)}")
    print(f"预期形状: (1, {enc['input_ids'].shape[1]}, 50257)  "
          f"(注意：Prefix Tuning 输出长度 = n，不含 p)")

    # 消融：去 reparam
    print("\n--- 消融对比：no reparam ---")
    m_no_reparam = PrefixTuningGPT2(prefix_length=10, use_reparam=False)
    print_param_summary(m_no_reparam, "PrefixTuningGPT2(no reparam, p=10)")


if __name__ == "__main__":
    main()
