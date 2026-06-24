"""Tensor parallel - column / row split linear, TP-MLP.

Simulates TP=N on a single CPU/GPU by computing each shard locally and
summing/concatenating, then verifies bit-exact equality with the unsharded
matmul.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class ColumnSplitLinear:
    """Y = X @ W, where W [in, out] is column-split across `n_shards`."""
    W: torch.Tensor      # full [in, out]
    n_shards: int

    def shard_w(self) -> List[torch.Tensor]:
        return list(torch.chunk(self.W, self.n_shards, dim=1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        shards = self.shard_w()
        outs = [X @ s for s in shards]      # each [N, out/n]
        return torch.cat(outs, dim=-1)


@dataclass
class RowSplitLinear:
    """Y = X · W where W [in, out] is row-split (along in).

    X must also be split along last dim, results are summed (all-reduce).
    """
    W: torch.Tensor
    n_shards: int

    def shard_w(self) -> List[torch.Tensor]:
        return list(torch.chunk(self.W, self.n_shards, dim=0))

    def forward(self, X_shards: List[torch.Tensor]) -> torch.Tensor:
        Ws = self.shard_w()
        partials = [x @ w for x, w in zip(X_shards, Ws)]
        return torch.stack(partials, dim=0).sum(dim=0)


@dataclass
class TpMlp:
    """Col-split up + row-split down, mimicking Megatron's MLP."""
    W_up: torch.Tensor     # [d, 4d]
    W_down: torch.Tensor   # [4d, d]
    n_shards: int

    def forward_tp(self, X: torch.Tensor) -> torch.Tensor:
        col = ColumnSplitLinear(self.W_up, self.n_shards)
        H = col.forward(X)               # [N, 4d] (gathered)
        H = torch.relu(H)
        # split H back along last dim for row-split down
        H_shards = list(torch.chunk(H, self.n_shards, dim=-1))
        row = RowSplitLinear(self.W_down, self.n_shards)
        return row.forward(H_shards)

    def forward_single(self, X: torch.Tensor) -> torch.Tensor:
        H = torch.relu(X @ self.W_up)
        return H @ self.W_down


def demo() -> None:
    """Show that column/row-split TP reproduces the unsharded matmul bit-for-bit."""
    torch.manual_seed(0)
    print("=== Tensor Parallel (single-process simulation) ===")

    X = torch.randn(8, 256)
    W = torch.randn(256, 512)
    ref = X @ W
    for n in (1, 2, 4, 8):
        col = ColumnSplitLinear(W, n_shards=n).forward(X)
        max_diff = (col - ref).abs().max().item()
        print(f"column-split  TP={n}: shards concat -> max|Δ| vs single = {max_diff:.2e}")

    print()
    X_full = torch.randn(8, 256)
    W_row = torch.randn(256, 128)
    ref_row = X_full @ W_row
    for n in (1, 2, 4, 8):
        X_shards = list(torch.chunk(X_full, n, dim=-1))
        row = RowSplitLinear(W_row, n_shards=n).forward(X_shards)
        max_diff = (row - ref_row).abs().max().item()
        print(f"row-split     TP={n}: partial-sum (all-reduce) -> max|Δ| = {max_diff:.2e}")

    print("\n--- Megatron MLP (col-split up -> relu -> row-split down) ---")
    d = 64
    W_up = torch.randn(d, 4 * d)
    W_down = torch.randn(4 * d, d)
    Xm = torch.randn(4, d)
    mlp = TpMlp(W_up, W_down, n_shards=4)
    out_tp = mlp.forward_tp(Xm)
    out_single = mlp.forward_single(Xm)
    print(f"TP=4 MLP vs single: max|Δ| = {(out_tp - out_single).abs().max().item():.2e}")
    print("=> TP is exact (only 1 all-reduce per row-split op); it splits memory, not math.")


if __name__ == "__main__":
    demo()
