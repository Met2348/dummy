"""(IA)³ — adapters 库调包版。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from adapters import AutoAdapterModel, AdapterConfig

sys.path.append(str(Path(__file__).parent))
from common import print_param_summary  # noqa: E402


def build_ia3_model_adapters():
    model = AutoAdapterModel.from_pretrained("gpt2")
    config = AdapterConfig.load("ia3")
    model.add_adapter("demo", config=config)
    model.train_adapter("demo")
    return model


def main() -> None:
    torch.manual_seed(42)
    model = build_ia3_model_adapters()
    print_param_summary(model, "adapters (IA)^3")


if __name__ == "__main__":
    main()
