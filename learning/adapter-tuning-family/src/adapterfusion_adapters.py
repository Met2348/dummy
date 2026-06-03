"""AdapterFusion — adapters 库调包版。

adapters 库 API:
    model.add_adapter("task_A", ...)
    model.add_adapter("task_B", ...)
    model.add_adapter("task_C", ...)
    # ... 假设单独训过了
    model.add_adapter_fusion(Fuse("task_A", "task_B", "task_C"))
    model.train_adapter_fusion(Fuse("task_A", "task_B", "task_C"))
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from adapters import AutoAdapterModel, AdapterConfig
from adapters.composition import Fuse

sys.path.append(str(Path(__file__).parent))
from common import print_param_summary  # noqa: E402


def build_fusion_model(n_adapters: int = 3, reduction_factor: int = 48):
    """构造 N 任务 fusion 模型（模拟：每个 adapter 都是新建的，没真训过）。"""
    model = AutoAdapterModel.from_pretrained("gpt2")
    config = AdapterConfig.load("pfeiffer", reduction_factor=reduction_factor)
    task_names = [f"task_{i}" for i in range(n_adapters)]
    for t in task_names:
        model.add_adapter(t, config=config)
    # 加 fusion
    fusion_setup = Fuse(*task_names)
    model.add_adapter_fusion(fusion_setup)
    model.train_adapter_fusion(fusion_setup)
    return model


def main() -> None:
    torch.manual_seed(42)
    model = build_fusion_model(n_adapters=3)
    print_param_summary(model, "adapters AdapterFusion (N=3)")


if __name__ == "__main__":
    main()
