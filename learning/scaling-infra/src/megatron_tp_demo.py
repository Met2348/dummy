"""Megatron-style Tensor Parallel - 单卡 mock 演示.

教学: 实现 ColumnLinear / RowLinear / MLP TP 的形状变化。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MockColumnLinear(nn.Module):
    """col-split: weight [d_in, d_out / tp]."""
    def __init__(self, d_in: int, d_out: int, tp_size: int = 1, tp_rank: int = 0):
        super().__init__()
        assert d_out % tp_size == 0
        self.d_out_local = d_out // tp_size
        self.W = nn.Parameter(torch.randn(d_in, self.d_out_local) * 0.01)

    def forward(self, x):
        return x @ self.W


class MockRowLinear(nn.Module):
    """row-split: weight [d_in / tp, d_out]."""
    def __init__(self, d_in: int, d_out: int, tp_size: int = 1, tp_rank: int = 0):
        super().__init__()
        assert d_in % tp_size == 0
        self.d_in_local = d_in // tp_size
        self.W = nn.Parameter(torch.randn(self.d_in_local, d_out) * 0.01)

    def forward(self, x_local):
        out = x_local @ self.W
        return out


class TpMlp(nn.Module):
    """TP MLP: col → activation → row → all-reduce."""
    def __init__(self, d: int, d_ff: int, tp_size: int = 1, tp_rank: int = 0):
        super().__init__()
        self.fc1 = MockColumnLinear(d, d_ff, tp_size, tp_rank)
        self.fc2 = MockRowLinear(d_ff, d, tp_size, tp_rank)

    def forward(self, x):
        h = self.fc1(x).relu()
        out = self.fc2(h)
        # 实际多卡: dist.all_reduce(out)
        return out


def gather_tp_outputs(local_outs: list) -> torch.Tensor:
    """模拟 tp all-reduce (sum)."""
    out = sum(local_outs)
    return out


if __name__ == "__main__":
    d, d_ff = 64, 256
    tp = 4

    print(f"=== TP MLP demo (d={d}, d_ff={d_ff}, tp={tp}) ===")
    print(f"Vanilla:   W1=[{d},{d_ff}], W2=[{d_ff},{d}]")
    print(f"TP:        W1=[{d},{d_ff//tp}] × {tp}, W2=[{d_ff//tp},{d}] × {tp}")

    mlps = [TpMlp(d, d_ff, tp_size=tp, tp_rank=i) for i in range(tp)]

    x = torch.randn(2, 10, d)
    local_outs = [mlps[i](x) for i in range(tp)]
    out = gather_tp_outputs(local_outs)
    print(f"\nOutput shape: {out.shape}")

    fc1_params = sum(p.numel() for p in mlps[0].fc1.parameters())
    fc2_params = sum(p.numel() for p in mlps[0].fc2.parameters())
    print(f"\nfc1 per-rank: {fc1_params} param  vs vanilla: {d * d_ff}")
    print(f"fc2 per-rank: {fc2_params} param  vs vanilla: {d * d_ff}")
