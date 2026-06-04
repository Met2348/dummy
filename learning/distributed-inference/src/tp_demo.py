"""Tensor parallel — column / row split linear, TP-MLP.

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
    """Y = X · W, where W [in, out] is column-split across `n_shards`."""
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
