"""Compacter — adapters 库调包版。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from adapters import AutoAdapterModel, AdapterConfig

sys.path.append(str(Path(__file__).parent))
from common import print_param_summary  # noqa: E402


def build_compacter_model(reduction_factor: int = 48, phm_dim: int = 4):
    """构造 Compacter 模型。

    adapters 库 config: "compacter" 或 "compacter++"
    """
    model = AutoAdapterModel.from_pretrained("gpt2")
    config = AdapterConfig.load("compacter", reduction_factor=reduction_factor, phm_dim=phm_dim)
    model.add_adapter("demo", config=config)
    model.train_adapter("demo")
    return model


def main() -> None:
    torch.manual_seed(42)
    model = build_compacter_model(reduction_factor=48, phm_dim=4)
    print_param_summary(model, "adapters Compacter (rf=48, n=4)")


if __name__ == "__main__":
    main()
