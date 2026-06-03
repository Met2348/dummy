"""Pfeiffer Adapter — adapters 库调包版。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from adapters import AutoAdapterModel, AdapterConfig

sys.path.append(str(Path(__file__).parent))
from common import print_param_summary  # noqa: E402


def build_pfeiffer_model(reduction_factor: int = 48, task_name: str = "demo"):
    model = AutoAdapterModel.from_pretrained("gpt2")
    config = AdapterConfig.load("pfeiffer", reduction_factor=reduction_factor)
    model.add_adapter(task_name, config=config)
    model.train_adapter(task_name)
    return model


def main() -> None:
    torch.manual_seed(42)
    model = build_pfeiffer_model(reduction_factor=48)
    print_param_summary(model, "adapters Pfeiffer (r=16, rf=48)")

    print("\n可训练参数（前 10 个）:")
    cnt = 0
    for name, p in model.named_parameters():
        if p.requires_grad and cnt < 10:
            print(f"  {name}: shape={tuple(p.shape)}")
            cnt += 1


if __name__ == "__main__":
    main()
