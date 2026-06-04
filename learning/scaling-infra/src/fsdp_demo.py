"""FSDP demo - 单进程模拟 + wrap policy 检查.

实际多卡需 torchrun --nproc_per_node=N.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                  nn.Linear(4 * d, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + a)
        x = self.norm2(x + self.mlp(x))
        return x


class TinyModel(nn.Module):
    def __init__(self, n_layer=4, d=128, vocab=1000):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([TransformerBlock(d)
                                     for _ in range(n_layer)])
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h)


def estimate_fsdp_memory(n_param: int, n_gpu: int = 4) -> dict:
    """估算 FSDP 各阶段显存."""
    bf16 = 2
    fp32 = 4
    return {
        "weights_shard_gb": n_param * bf16 / n_gpu / 1e9,
        "grad_shard_gb": n_param * bf16 / n_gpu / 1e9,
        "optimizer_shard_gb": n_param * fp32 * 2 / n_gpu / 1e9,
        "all_gather_peak_gb": n_param * bf16 / 1e9,
    }


def print_fsdp_setup():
    """打印 FSDP setup 模板."""
    print("=== FSDP 标准 setup ===")
    print("""
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision, ShardingStrategy, BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
import functools

dist.init_process_group("nccl")
model = MyModel()

policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},
)

mp = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=policy,
    mixed_precision=mp,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    limit_all_gathers=True,
)
""")


if __name__ == "__main__":
    m = TinyModel(n_layer=4, d=128, vocab=1000)
    n = sum(p.numel() for p in m.parameters())
    print(f"TinyModel: {n/1e6:.2f}M params")

    for ngpu in [1, 4, 8, 16]:
        r = estimate_fsdp_memory(n_param=8e9, n_gpu=ngpu)
        print(f"\n=== Llama-8B FSDP on {ngpu} GPU ===")
        for k, v in r.items():
            print(f"  {k}: {v:.2f}")

    print_fsdp_setup()
