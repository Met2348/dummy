"""Pipeline parallel mock + bubble estimator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List


@dataclass
class PipelineStage:
    layer_fns: List[Callable[[object], object]]

    def forward(self, h):
        for fn in self.layer_fns:
            h = fn(h)
        return h


def gpipe_bubble(n_stages: int, n_micro: int) -> float:
    """Bubble fraction for naive GPipe schedule."""
    busy = n_stages * n_micro
    total = busy + (n_stages - 1) * 2
    return (total - busy) / total


def interleaved_bubble(n_stages: int, n_micro: int, n_chunks: int) -> float:
    """Bubble fraction with interleaved 1F1B."""
    effective_stages = n_stages / n_chunks
    return (effective_stages - 1) / (effective_stages - 1 + n_micro)


def schedule_naive(n_stages: int, n_micro: int) -> List[List[str]]:
    """Return a stage-by-time schedule grid."""
    grid: List[List[str]] = [["·"] * (n_stages + n_micro - 1) for _ in range(n_stages)]
    for m in range(n_micro):
        for s in range(n_stages):
            grid[s][m + s] = f"F{m + 1}"
    return grid


def render_grid(grid: List[List[str]]) -> str:
    return "\n".join("|" + " ".join(f"{c:>3}" for c in row) + "|" for row in grid)
