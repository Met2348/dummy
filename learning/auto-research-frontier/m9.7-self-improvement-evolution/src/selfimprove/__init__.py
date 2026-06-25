"""selfimprove：自我改进的进化档案循环，演示 fitness 被 game（reward hacking）。"""
from .genome import (
    HOLDOUT, LEAKED, SEED, Genome, holdout_fitness, naive_fitness, true_label,
)
from .evolve import evolve

__all__ = [
    "HOLDOUT", "LEAKED", "SEED", "Genome", "holdout_fitness", "naive_fitness",
    "true_label", "evolve",
]
