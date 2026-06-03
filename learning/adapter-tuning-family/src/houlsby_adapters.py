"""Houlsby Adapter — AdapterHub `adapters` 库调包版。

用 adapters.AdapterConfig.load("houlsby") 加载预定义配置。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from adapters import AutoAdapterModel, AdapterConfig
from transformers import GPT2Tokenizer

sys.path.append(str(Path(__file__).parent))
from common import print_param_summary  # noqa: E402


def build_houlsby_model(reduction_factor: int = 48, task_name: str = "demo"):
    """构造 Houlsby adapter 模型。

    adapters 库用 reduction_factor 代替 r:
        r = d / reduction_factor
        GPT-2 d=768, reduction_factor=48 → r=16
    """
    model = AutoAdapterModel.from_pretrained("gpt2")
    config = AdapterConfig.load("houlsby", reduction_factor=reduction_factor)
    model.add_adapter(task_name, config=config)
    model.train_adapter(task_name)
    return model


def main() -> None:
    torch.manual_seed(42)
    model = build_houlsby_model(reduction_factor=48)
    print_param_summary(model, "adapters Houlsby (r=16, rf=48)")

    print("\n可训练参数（前 10 个）:")
    cnt = 0
    for name, p in model.named_parameters():
        if p.requires_grad and cnt < 10:
            print(f"  {name}: shape={tuple(p.shape)}")
            cnt += 1


if __name__ == "__main__":
    main()
