"""K-Adapter — adapters 库调包版（用 Stack composition 模拟）。

adapters 库没有原生 K-Adapter；用多个 adapter + Stack 模拟"知识叠加"。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from adapters import AutoAdapterModel, AdapterConfig
from adapters.composition import Stack

sys.path.append(str(Path(__file__).parent))
from common import print_param_summary  # noqa: E402


def build_k_adapter_model(reduction_factor: int = 48):
    """构造 K-Adapter 风格的模型。

    用 Stack 组合多个 adapter 模拟"knowledge stacking"。
    """
    model = AutoAdapterModel.from_pretrained("gpt2")
    config = AdapterConfig.load("pfeiffer", reduction_factor=reduction_factor)
    model.add_adapter("factual", config=config)
    model.add_adapter("linguistic", config=config)
    # Stack: 顺序激活 (≈ K-Adapter 的"知识层叠")
    model.active_adapters = Stack("factual", "linguistic")
    model.train_adapter(["factual", "linguistic"])
    return model


def main() -> None:
    torch.manual_seed(42)
    model = build_k_adapter_model()
    print_param_summary(model, "adapters K-Adapter (factual + linguistic stack)")


if __name__ == "__main__":
    main()
