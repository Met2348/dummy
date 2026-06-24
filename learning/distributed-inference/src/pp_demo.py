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


def demo() -> None:
    """Show the pipeline bubble shrinking with micro-batches and with 1F1B."""
    n_stages = 4
    print("=== Pipeline Parallel: bubble fraction ===")
    print(f"stages = {n_stages}\n")
    print(f"{'n_micro':>8}{'GPipe bubble':>15}{'1F1B(2 chunks)':>17}")
    for n_micro in (1, 2, 4, 8, 16, 32):
        g = gpipe_bubble(n_stages, n_micro)
        i = interleaved_bubble(n_stages, n_micro, n_chunks=2)
        print(f"{n_micro:>8}{g:>14.1%}{i:>16.1%}")
    print("\n=> more micro-batches -> smaller bubble; interleaved 1F1B shrinks it further.")

    print(f"\n--- naive GPipe schedule grid (stages={n_stages}, micro=4) ---")
    print("rows = pipeline stages, cols = time steps, F<m> = forward of micro-batch m")
    print(render_grid(schedule_naive(n_stages, 4)))
    print("(leading/trailing '·' cells are the bubble: idle stages while the pipe fills/drains.)")


if __name__ == "__main__":
    demo()
