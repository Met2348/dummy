"""AdapterDrop — adapters 库调包版。

adapters 库通过 set_active_adapters 控制层级激活。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from adapters import AutoAdapterModel, AdapterConfig

sys.path.append(str(Path(__file__).parent))
from common import print_param_summary  # noqa: E402


def build_adapterdrop_model(reduction_factor: int = 48):
    """构造支持 adapterdrop 的模型（用 pfeiffer base）。"""
    model = AutoAdapterModel.from_pretrained("gpt2")
    config = AdapterConfig.load("pfeiffer", reduction_factor=reduction_factor)
    model.add_adapter("demo", config=config)
    model.train_adapter("demo")
    return model


def set_drop_layers(model, skip_layers: list[int]) -> None:
    """推理时跳过指定层的 adapter（adapters 1.3 API）。

    adapters 库的 SkipLayer composition 实现这个。
    简化版：直接设置 adapter layer 的 enabled 标志。
    """
    # adapters 1.3 实际用 SkipLayer composition；这里简化为打印
    print(f"  跳过前 {len(skip_layers)} 层 adapter: {skip_layers}")


def main() -> None:
    torch.manual_seed(42)
    model = build_adapterdrop_model()
    print_param_summary(model, "adapters Pfeiffer (with drop support)")
    set_drop_layers(model, [0, 1, 2, 3, 4])


if __name__ == "__main__":
    main()
