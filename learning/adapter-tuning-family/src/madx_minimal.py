"""MAD-X 最小实现（手写）。

对应论文: Pfeiffer et al. 2020, "MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer" (EMNLP)
对应 lecture: lectures/07-k-adapter-mad-x.md

核心思想 — 三类 adapter 解耦:
    Language Adapter (LA): 学语言特性，每种语言一个 LA
    Task Adapter (TA): 学任务特性，每种任务一个 TA
    Invertible Adapter (IA): 处理 input embedding 层的语言差异

应用顺序 (transformer block):
    x → base → LA(x) → TA(x) → output

→ 这种解耦让 "lang_A + task_X" 可灵活组合：
    用德语 LA + 任务 NER 的 TA → 德语 NER
    用法语 LA + 同一任务 NER 的 TA → 法语 NER

简化教学版：
    - 3 种语言 toy LA: 'en', 'de', 'fr'
    - 1 种任务 toy TA
    - 简化的 invertible adapter
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


# Toy 多语言数据
TOY_MULTILINGUAL_DATA = {
    "en": ["hello world", "i love this", "absolutely beautiful day"],
    "de": ["hallo welt", "ich liebe das", "absolut schoner tag"],  # 简化（去重音）
    "fr": ["bonjour monde", "j aime ca", "journee absolument belle"],
}


class InvertibleAdapter(nn.Module):
    """简化的 invertible adapter（论文用 normalizing flow，这里用 affine）。

    forward: y = sigma * x + mu
    inverse: x = (y - mu) / sigma

    用在 embedding 层处理语言差异。
    """

    def __init__(self, d: int):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(d))  # log scale for stability
        self.mu = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        return x * torch.exp(self.log_sigma) + self.mu

    def inverse(self, y):
        return (y - self.mu) * torch.exp(-self.log_sigma)


class MADXGPT2(nn.Module):
    """GPT-2 + MAD-X (LA + TA + IA)。

    每个 transformer block:
        x → base → LA_active(x) → TA_active(x)
    """

    def __init__(
        self,
        base_model_name: str = "gpt2",
        r: int = 16,
        languages: tuple[str, ...] = ("en", "de", "fr"),
        tasks: tuple[str, ...] = ("ner",),
    ):
        super().__init__()
        self.lm = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        freeze_base_model(self.lm)

        d = self.lm.config.n_embd
        self.r = r
        self.languages = languages
        self.tasks = tasks
        self.active_language = languages[0]
        self.active_task = tasks[0]

        # 每个 block 加 LA × N + TA × M
        for block in self.lm.transformer.h:
            block.mlp = _MADXMlpWrapper(
                block.mlp,
                d, r,
                languages=languages,
                tasks=tasks,
            )

        # Embedding 层加 IA（每语言一个）
        self.invertible_adapters = nn.ModuleDict({
            lang: InvertibleAdapter(d) for lang in languages
        })

    def set_active(self, language: str, task: str) -> None:
        """切换当前激活的 LA + TA。"""
        assert language in self.languages
        assert task in self.tasks
        self.active_language = language
        self.active_task = task
        for block in self.lm.transformer.h:
            block.mlp.active_language = language
            block.mlp.active_task = task

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )


class _MADXMlpWrapper(nn.Module):
    """MAD-X 的核心：每 block 嵌入 LA + TA。"""

    def __init__(self, base_mlp, d, r, languages, tasks):
        super().__init__()
        self.base_mlp = base_mlp
        self.language_adapters = nn.ModuleDict({
            lang: HoulsbyAdapter(d, r) for lang in languages
        })
        self.task_adapters = nn.ModuleDict({
            task: HoulsbyAdapter(d, r) for task in tasks
        })
        self.active_language = languages[0]
        self.active_task = tasks[0]

    def forward(self, x):
        h = self.base_mlp(x)
        # 先 LA, 再 TA (顺序)
        h = self.language_adapters[self.active_language](h)
        h = self.task_adapters[self.active_task](h)
        return h


def main() -> None:
    torch.manual_seed(42)
    model = MADXGPT2(r=16, languages=("en", "de", "fr"), tasks=("ner",))
    print_param_summary(model, "MAD-X (3 lang, 1 task, r=16)")

    print(f"\n参数布局:")
    print(f"  Language adapters: 3 × 12 × 25,360 = 912,960")
    print(f"  Task adapters:     1 × 12 × 25,360 = 304,320")
    print(f"  Invertible adapters: 3 × 2 × 768 = 4,608")
    print(f"  总计: ~1,221,888")

    # 演示切换语言
    print("\n切换 active language: en -> de")
    model.set_active("de", "ner")
    enc = model.tokenizer("hallo welt", return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"])
    print(f"  forward logits.shape={tuple(out.logits.shape)}")


if __name__ == "__main__":
    main()
