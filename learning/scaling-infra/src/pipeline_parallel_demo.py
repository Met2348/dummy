"""Pipeline Parallel - bubble vs micro-batch 数演示."""
from __future__ import annotations

import torch
import torch.nn as nn


def gpipe_bubble(n_stage: int, n_micro: int) -> float:
    """GPipe / 1F1B bubble 占比."""
    return (n_stage - 1) / (n_stage + n_micro - 1)


def interleaved_bubble(n_stage: int, n_micro: int, n_chunk: int = 4) -> float:
    """interleaved 1F1B."""
    return (n_stage - 1) / (n_chunk * (n_stage + n_micro - 1))


class PpStage(nn.Module):
    """单个 PP stage."""
    def __init__(self, layers: list):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for L in self.layers:
            x = L(x)
        return x


def split_model_into_stages(layers: list, n_stage: int) -> list:
    """将模型 layers 切成 n_stage 段."""
    per = len(layers) // n_stage
    stages = []
    for i in range(n_stage):
        start = i * per
        end = (i + 1) * per if i < n_stage - 1 else len(layers)
        stages.append(PpStage(layers[start:end]))
    return stages


if __name__ == "__main__":
    print("=== PP bubble vs micro-batch (8 stage) ===")
    for M in [1, 4, 8, 16, 32, 64, 128]:
        b1 = gpipe_bubble(8, M)
        b2 = interleaved_bubble(8, M, n_chunk=4)
        print(f"  M={M:>3}  GPipe/1F1B={b1:.1%}  Interleaved={b2:.1%}")

    print("\n=== Stage 切分 demo ===")
    layers = [nn.Linear(64, 64) for _ in range(24)]
    stages = split_model_into_stages(layers, n_stage=8)
    for i, s in enumerate(stages):
        print(f"  stage {i}: {len(s.layers)} layers")
