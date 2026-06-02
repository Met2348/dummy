"""
Prompt Tuning 最小实现（手写，不依赖 peft）。

对应论文：Lester et al., 2021, arXiv:2104.08691
对应 lecture: lectures/02-prompt-tuning.md

核心思想：冻结 GPT-2，只训练一段长度为 p 的 prompt embedding。

公式索引（与 lecture 对齐）：
    Ẽ = [P; E]                                       (1)
    H = Transformer(Ẽ; θ_LM)                         (2)
    P(y_t | y_<t, Ẽ) = softmax(W_out h_{p+n+t-1})    (3)
    φ* = argmin -Σ log P(y_t | y_<t, Ẽ(x))           (4)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent))
from common import ToySentimentDataset, print_param_summary  # noqa: E402


class PromptTuningGPT2(nn.Module):
    """Prompt Tuning 包装器：冻结 GPT-2，仅训练 prompt_embeddings (p, d)。

    Args:
        base_model_name: HF 模型名（默认 gpt2，117M）
        prompt_length:   p，可训练 prompt 的长度
        init_text:       若提供，把该段文本的 token embedding 作为初始化
                         （class label init 策略；省略则用 N(0, 0.02)）
    """

    def __init__(
        self,
        base_model_name: str = "gpt2",
        prompt_length: int = 10,
        init_text: str | None = None,
    ):
        super().__init__()
        self.lm = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 公式 (4) 中 θ_LM 全部冻结
        for p in self.lm.parameters():
            p.requires_grad = False

        self.prompt_length = prompt_length          # p
        self.embed_dim = self.lm.config.n_embd      # d (GPT-2 base = 768)

        # 公式 (1) 中可训练的 P ∈ R^(p, d)
        self.prompt_embeddings = nn.Parameter(
            torch.empty(prompt_length, self.embed_dim)
        )
        self._init_prompt(init_text)

    def _init_prompt(self, init_text: str | None) -> None:
        """初始化 P 的两种策略：随机正态 / 用真实词的 embedding。"""
        if init_text is None:
            # Random init: N(0, 0.02^2)（论文小模型上效果差，大模型上够用）
            nn.init.normal_(self.prompt_embeddings, mean=0.0, std=0.02)
        else:
            # Class-label-style init: 复用真实 token embedding
            ids = self.tokenizer(init_text, return_tensors="pt")["input_ids"][0]
            # 重复或截断到 prompt_length
            n_repeat = self.prompt_length // len(ids) + 1
            ids = ids.repeat(n_repeat)[: self.prompt_length]
            with torch.no_grad():
                emb = self.lm.transformer.wte(ids).clone()
            self.prompt_embeddings.data.copy_(emb)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: torch.LongTensor | None = None,
    ):
        """前向：实现公式 (1) → (2) → (3)。

        Args:
            input_ids:      (B, n)
            attention_mask: (B, n)
            labels:         (B, n) 或 None

        Returns:
            CausalLMOutputWithCrossAttentions, 含 logits (B, p+n, V) 与 loss。
        """
        B = input_ids.shape[0]

        # 公式 (1): 拼接
        token_embeds = self.lm.transformer.wte(input_ids)            # E:  (B, n, d)
        prompt = self.prompt_embeddings.unsqueeze(0).expand(B, -1, -1)  # P扩展: (B, p, d)
        inputs_embeds = torch.cat([prompt, token_embeds], dim=1)     # Ẽ:  (B, p+n, d)

        # attention mask: prompt 位置恒为可见（前 p 位置 = 1）
        prompt_mask = torch.ones(
            B, self.prompt_length, dtype=attention_mask.dtype, device=attention_mask.device
        )
        full_mask = torch.cat([prompt_mask, attention_mask], dim=1)  # (B, p+n)

        # labels: prompt 位置不参与 loss 计算，填 -100
        full_labels = labels
        if labels is not None:
            prompt_labels = torch.full(
                (B, self.prompt_length), -100, dtype=labels.dtype, device=labels.device
            )
            full_labels = torch.cat([prompt_labels, labels], dim=1)  # (B, p+n)

        # 公式 (2) + (3): 走完整 LM（θ_LM 冻结，梯度只会到 P）
        return self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=full_labels,
        )


def toy_train(model: PromptTuningGPT2, num_steps: int = 20) -> None:
    """Toy 训练循环：证明梯度只流到 prompt_embeddings。

    用 input_ids 自身作 label，演示语言建模 loss。
    """
    ds = ToySentimentDataset(model.tokenizer, max_len=16)
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    optim = torch.optim.AdamW([model.prompt_embeddings], lr=1e-2)

    model.train()
    step = 0
    while step < num_steps:
        for batch in loader:
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"],
            )
            optim.zero_grad()
            out.loss.backward()
            optim.step()
            step += 1
            print(f"step={step}, loss={out.loss.item():.4f}")
            if step >= num_steps:
                break


def main() -> None:
    torch.manual_seed(42)
    model = PromptTuningGPT2(prompt_length=10)
    print_param_summary(model, "PromptTuningGPT2(prompt_length=10)")
    # Expected: trainable=10*768=7,680 / total=124,447,872 (0.0062%)

    # 前向冒烟测试
    enc = model.tokenizer("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"])
    n_in = enc["input_ids"].shape[1]
    print(f"\n前向输出 logits.shape={tuple(out.logits.shape)}")
    print(f"预期 = (1, {model.prompt_length}+{n_in}={model.prompt_length + n_in}, vocab=50257)")

    # Toy 训练
    print("\n开始 toy 训练（5 步）：")
    toy_train(model, num_steps=5)


if __name__ == "__main__":
    main()
