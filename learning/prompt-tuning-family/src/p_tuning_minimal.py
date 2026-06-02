"""
P-Tuning v1 最小实现（手写）。

对应论文：Liu et al., 2021, arXiv:2103.10385
对应 lecture: lectures/03-p-tuning.md

核心思想：
  - 冻结 GPT-2
  - 在输入层加 p 个 pseudo token 的 embedding
  - 关键：用 Bi-LSTM + MLP 做 reparameterization 生成 prompt
  - 仅训练 prompt encoder（Embedding + LSTM + MLP），LM 不动

公式索引：
    T = [pre_0] ⊕ [x_S1] ⊕ ... ⊕ [pre_k] ⊕ [x_S{k+1}]                     (1)
    (h_0, ..., h_{p-1}) = MLP_φ(BiLSTM_φ(u_0, ..., u_{p-1}))               (2)
    E_T = (h_0, ..., h_{p-1}, e(x_anchor), e(x_1), ...)                    (3)
    φ* = argmin L(LM_θ(E_T), y)                                            (4)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent))
from common import print_param_summary  # noqa: E402


class PromptEncoder(nn.Module):
    """公式 (2) 的实现：Bi-LSTM + MLP，生成 p 个 pseudo token 的 embedding。"""

    def __init__(self, prompt_length: int, embed_dim: int, hidden: int = 256):
        super().__init__()
        # 公式 (2) 中的 {u_i}_i=0^(p-1)
        self.embed = nn.Embedding(prompt_length, embed_dim)
        # 公式 (2) 中的 BiLSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        # 公式 (2) 中的 MLP: Linear(2h, 2h) → ReLU → Linear(2h, d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, embed_dim),
        )
        self.prompt_length = prompt_length

    def forward(self) -> torch.Tensor:
        """返回 (p, d) 的 prompt embedding。"""
        device = next(self.parameters()).device
        ids = torch.arange(self.prompt_length, device=device).unsqueeze(0)  # (1, p)
        u = self.embed(ids)            # (1, p, d)  ← {u_i}
        lstm_out, _ = self.lstm(u)     # (1, p, 2h)
        return self.mlp(lstm_out).squeeze(0)  # (p, d)  ← {h_i}


class PTuningGPT2(nn.Module):
    """P-Tuning v1 包装器（简化版：prompt 拼到输入前面，不支持任意位置）。"""

    def __init__(
        self,
        base_model_name: str = "gpt2",
        prompt_length: int = 10,
        encoder_hidden: int = 256,
    ):
        super().__init__()
        self.lm = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 公式 (4): θ_LM 冻结
        for p in self.lm.parameters():
            p.requires_grad = False

        self.prompt_length = prompt_length          # p
        self.embed_dim = self.lm.config.n_embd      # d
        self.encoder = PromptEncoder(
            prompt_length=prompt_length,
            embed_dim=self.embed_dim,
            hidden=encoder_hidden,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: torch.LongTensor | None = None,
    ):
        """前向：公式 (3) → (4)。

        Args:
            input_ids:      (B, n)  文本 token（包含 anchor 时直接在 input_ids 里）
            attention_mask: (B, n)
            labels:         (B, n) 或 None

        Returns:
            logits: (B, p+n, V)
        """
        B = input_ids.shape[0]

        # 公式 (2)→(3): 生成 prompt embedding
        prompt_embeds = self.encoder()                                  # (p, d)
        prompt_embeds = prompt_embeds.unsqueeze(0).expand(B, -1, -1)    # (B, p, d)

        # 公式 (3): 拼接 [prompt; text_tokens]
        token_embeds = self.lm.transformer.wte(input_ids)              # (B, n, d)
        inputs_embeds = torch.cat([prompt_embeds, token_embeds], dim=1)  # (B, p+n, d)

        # 扩展 mask（prompt 部分恒可见）
        prompt_mask = torch.ones(
            B, self.prompt_length, dtype=attention_mask.dtype, device=attention_mask.device
        )
        full_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # labels: prompt 位填 -100
        if labels is not None:
            prompt_labels = torch.full(
                (B, self.prompt_length), -100, dtype=labels.dtype, device=labels.device
            )
            labels = torch.cat([prompt_labels, labels], dim=1)

        # 公式 (4): 走完整 LM（冻结）
        return self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=labels,
        )


def main() -> None:
    torch.manual_seed(42)
    model = PTuningGPT2(prompt_length=10, encoder_hidden=256)
    print_param_summary(model, "PTuningGPT2(p=10, h=256)")

    # 拆解参数量来源
    n_embed = model.encoder.embed.weight.numel()
    n_lstm = sum(p.numel() for p in model.encoder.lstm.parameters())
    n_mlp = sum(p.numel() for p in model.encoder.mlp.parameters())
    print(f"\n参数分解：")
    print(f"  embedding (u_i):     {n_embed:,}")
    print(f"  Bi-LSTM:             {n_lstm:,}")
    print(f"  MLP head:            {n_mlp:,}")
    print(f"  合计:                {n_embed + n_lstm + n_mlp:,}")

    enc = model.tokenizer("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"])
    print(f"\n前向输出 logits.shape={tuple(out.logits.shape)}")


if __name__ == "__main__":
    main()
