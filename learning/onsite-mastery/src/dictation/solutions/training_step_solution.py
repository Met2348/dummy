"""training_step 参考实现——见 dictation/specs/training_step_spec.py 的接口约定。
风格对齐 interview-prep/src/mlcoding/training_loop.py 里 train() 内层循环的四步。
"""
from __future__ import annotations

from typing import Callable

import torch


def training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    clip_norm: float,
) -> float:
    x, y = batch
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    optimizer.step()
    return loss.item()


if __name__ == "__main__":
    from dictation.checks.training_step_check import check
    check(training_step)
    print("[PASS] training_step_solution 自检通过")
