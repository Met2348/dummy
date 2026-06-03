"""MAD-X — adapters 库调包版。

adapters 库原生支持 MAD-X via Stack(lang_adapter, task_adapter)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from adapters import AutoAdapterModel, AdapterConfig
from adapters.composition import Stack

sys.path.append(str(Path(__file__).parent))
from common import print_param_summary  # noqa: E402


def build_madx_model(reduction_factor: int = 48):
    """构造 MAD-X 风格的模型（lang + task stack）。"""
    model = AutoAdapterModel.from_pretrained("gpt2")
    config = AdapterConfig.load("pfeiffer", reduction_factor=reduction_factor)
    # 3 lang adapter + 1 task adapter
    for lang in ["en", "de", "fr"]:
        model.add_adapter(f"lang_{lang}", config=config)
    model.add_adapter("task_ner", config=config)
    # Stack: lang -> task 顺序激活
    model.active_adapters = Stack("lang_en", "task_ner")
    model.train_adapter(["lang_en", "lang_de", "lang_fr", "task_ner"])
    return model


def main() -> None:
    torch.manual_seed(42)
    model = build_madx_model()
    print_param_summary(model, "adapters MAD-X (3 lang + 1 task)")


if __name__ == "__main__":
    main()
