"""K-Adapter 最小实现（手写）。

对应论文: Wang et al. 2020, "K-Adapter: Infusing Knowledge into Pre-Trained Models with Adapters" (ACL findings)
对应 lecture: lectures/07-k-adapter-mad-x.md

核心思想：
    把"知识"封装到 adapter，让 PLM 接入多种领域知识但保持基座完整

架构:
    主干: 冻结 PLM (BERT/GPT-2)
    多个并行 K-Adapter:
        - factual adapter (训于 Wikidata triples)
        - linguistic adapter (训于 dependency parsing)
        - ...

    在某些层 (通常 0, 11, 22) 把 hidden 送进 adapter，
    adapter 输出 concat 回主干 hidden

每个 K-Adapter:
    multi-layer Transformer block + projection
    比 Houlsby Adapter 更复杂（不只是 down/up）

简化教学版:
    用 down/up 结构，但区分"factual" vs "linguistic" 两类
    用 toy 知识三元组 (10 条) 训练 factual adapter
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


# Toy 知识三元组（用于教学：模拟 Wikidata）
TOY_FACTUAL_TRIPLES = [
    "paris is the capital of france",
    "berlin is the capital of germany",
    "tokyo is the capital of japan",
    "rome is the capital of italy",
    "london is the capital of uk",
    "madrid is the capital of spain",
    "moscow is the capital of russia",
    "beijing is the capital of china",
    "ottawa is the capital of canada",
    "canberra is the capital of australia",
]

TOY_LINGUISTIC_DATA = [
    "the cat sat on the mat",
    "dogs chase cats around the yard",
    "she quickly opened the door",
    "they were singing beautifully",
    "running rapidly through the forest",
]


class KAdapter(nn.Module):
    """单个 K-Adapter：在特定层注入领域知识。

    与 Houlsby 相同的 bottleneck 结构，但语义上代表"某类知识"。
    """

    def __init__(self, d: int, r: int = 16, knowledge_type: str = "generic"):
        super().__init__()
        self.adapter = HoulsbyAdapter(d, r)
        self.knowledge_type = knowledge_type

    def forward(self, x):
        return self.adapter(x)


class MultiKnowledgeGPT2(nn.Module):
    """GPT-2 + 多个 K-Adapter（多类知识并存）。

    每个 transformer block 都挂多个 K-Adapter，输出加和:
        out = base(x) + adapter_factual(x) + adapter_linguistic(x)
    """

    def __init__(
        self,
        base_model_name: str = "gpt2",
        r: int = 16,
        knowledge_types: tuple[str, ...] = ("factual", "linguistic"),
    ):
        super().__init__()
        self.lm = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        freeze_base_model(self.lm)

        d = self.lm.config.n_embd
        self.r = r
        self.knowledge_types = knowledge_types

        for block in self.lm.transformer.h:
            adapters = nn.ModuleList([
                KAdapter(d, r, kt) for kt in knowledge_types
            ])
            block.mlp = _KMlpWrapper(block.mlp, adapters)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def freeze_adapter(self, knowledge_type: str) -> None:
        """冻结指定类别的 adapter（模拟 K-Adapter "训完冻结"）。"""
        for block in self.lm.transformer.h:
            for ka in block.mlp.adapters:
                if ka.knowledge_type == knowledge_type:
                    for p in ka.parameters():
                        p.requires_grad = False


class _KMlpWrapper(nn.Module):
    """挂载多个 K-Adapter，输出相加。"""

    def __init__(self, base_mlp, adapters: nn.ModuleList):
        super().__init__()
        self.base_mlp = base_mlp
        self.adapters = adapters

    def forward(self, x):
        h = self.base_mlp(x)
        # 每个 adapter 各算一次，输出相加（K-Adapter 的"加法组合"）
        delta = sum(a(h) - h for a in self.adapters)  # 取增量
        return h + delta


def main() -> None:
    torch.manual_seed(42)
    model = MultiKnowledgeGPT2(r=16, knowledge_types=("factual", "linguistic"))
    print_param_summary(model, "K-Adapter (2 类知识, r=16)")
    # 每 block 2 个 adapter = 2 × 25,360 = 50,720
    # 12 layer × 50,720 = 608,640
    print(f"\n参数布局: 2 类 × 12 layer × 25,360 = 608,640")

    # 演示冻结 factual adapter
    print("\n冻结 factual adapter:")
    model.freeze_adapter("factual")
    after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  冻结后可训练: {after:,} (应为只剩 linguistic 的一半)")


if __name__ == "__main__":
    main()
