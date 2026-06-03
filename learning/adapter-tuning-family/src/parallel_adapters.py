"""Parallel Adapter — adapters 库调包版。

adapters 库 config: "scaled_parallel"
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from adapters import AutoAdapterModel, AdapterConfig

sys.path.append(str(Path(__file__).parent))
from common import print_param_summary  # noqa: E402


def build_parallel_model(reduction_factor: int = 48):
    """构造 Parallel Adapter 模型。"""
    model = AutoAdapterModel.from_pretrained("gpt2")
    config = AdapterConfig.load("parallel", reduction_factor=reduction_factor)
    model.add_adapter("demo", config=config)
    model.train_adapter("demo")
    return model


def main() -> None:
    torch.manual_seed(42)
    model = build_parallel_model(reduction_factor=48)
    print_param_summary(model, "adapters Parallel (rf=48)")


if __name__ == "__main__":
    main()
