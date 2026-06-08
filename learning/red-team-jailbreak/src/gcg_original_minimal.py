"""Safe toy version of Greedy Coordinate Gradient from the GCG paper.

The real paper optimizes suffix tokens against aligned LLMs. This file does
not contain harmful prompts, real model calls, or reusable jailbreak strings.
It only demonstrates the discrete optimization shape on a tiny differentiable
toy objective: make a suffix score a harmless SAFE_ACK class.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


VOCAB = [
    "<pad>",
    "<toy_a>",
    "<toy_b>",
    "<toy_c>",
    "<toy_d>",
    "<toy_e>",
    "<toy_f>",
    "<safe_ack>",
]


@dataclass(frozen=True)
class GCGStepTrace:
    """One accepted coordinate replacement."""

    position: int
    old_token: str
    new_token: str
    old_loss: float
    new_loss: float


def toy_token_effects() -> torch.Tensor:
    """Return per-position token scores for a harmless target.

    Shape is (suffix_len, vocab_size). Larger score means the suffix is more
    likely to produce the toy SAFE_ACK label.
    """

    return torch.tensor(
        [
            [0.0, 0.1, 0.2, -0.3, 0.4, 0.0, 0.3, 1.5],
            [0.0, 0.3, -0.2, 0.2, 0.1, 1.2, 0.4, 0.6],
            [0.0, -0.1, 1.0, 0.2, 0.3, 0.0, 0.5, 0.7],
        ],
        dtype=torch.float32,
    )


def suffix_to_one_hot(suffix: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Convert token ids with shape (L,) to one-hot shape (L, V)."""

    return torch.nn.functional.one_hot(suffix, num_classes=vocab_size).float()


def toy_loss_from_one_hot(one_hot: torch.Tensor, effects: torch.Tensor) -> torch.Tensor:
    """Negative harmless target score.

    This is deliberately simple. It lets the gradient with respect to one-hot
    token choices indicate which replacements could reduce loss.
    """

    return -(one_hot * effects).sum()


def exact_loss(suffix: torch.Tensor, effects: torch.Tensor) -> torch.Tensor:
    """Evaluate the toy loss for a discrete suffix."""

    one_hot = suffix_to_one_hot(suffix, effects.shape[1])
    return toy_loss_from_one_hot(one_hot, effects)


def gcg_step(suffix: torch.Tensor, effects: torch.Tensor, top_k: int = 2) -> tuple[torch.Tensor, GCGStepTrace]:
    """Run one safe GCG coordinate step.

    This mirrors the paper's high-level loop:
    1. Compute gradients with respect to one-hot token indicators.
    2. For every position, keep the top-k negative-gradient token candidates.
    3. Evaluate candidate replacements exactly with a forward pass.
    4. Accept the candidate with the lowest loss.
    """

    one_hot = suffix_to_one_hot(suffix, effects.shape[1])
    one_hot.requires_grad_(True)
    old_loss = toy_loss_from_one_hot(one_hot, effects)
    old_loss.backward()

    candidates: list[tuple[float, int, int, torch.Tensor]] = []
    for pos in range(suffix.numel()):
        ranked = torch.topk(-one_hot.grad[pos], k=top_k).indices
        for token_id in ranked.tolist():
            trial = suffix.clone()
            trial[pos] = token_id
            loss = exact_loss(trial, effects)
            candidates.append((float(loss), pos, token_id, trial))

    best_loss, best_pos, best_token, best_suffix = min(candidates, key=lambda item: item[0])
    trace = GCGStepTrace(
        position=best_pos,
        old_token=VOCAB[int(suffix[best_pos])],
        new_token=VOCAB[best_token],
        old_loss=float(old_loss.detach()),
        new_loss=best_loss,
    )
    return best_suffix, trace


def optimize_suffix(initial_suffix: torch.Tensor, steps: int = 4, top_k: int = 2) -> tuple[torch.Tensor, list[GCGStepTrace]]:
    """Optimize a toy suffix with repeated safe GCG steps."""

    effects = toy_token_effects()
    suffix = initial_suffix.clone()
    traces: list[GCGStepTrace] = []
    for _ in range(steps):
        new_suffix, trace = gcg_step(suffix, effects, top_k=top_k)
        traces.append(trace)
        if torch.equal(new_suffix, suffix):
            break
        suffix = new_suffix
    return suffix, traces


def _self_test() -> int:
    initial = torch.tensor([0, 0, 0])
    effects = toy_token_effects()
    final, traces = optimize_suffix(initial, steps=4, top_k=3)

    assert traces
    assert exact_loss(final, effects) < exact_loss(initial, effects)
    assert VOCAB[int(final[0])] == "<safe_ack>"
    assert VOCAB[int(final[1])] == "<toy_e>"
    assert VOCAB[int(final[2])] == "<toy_b>"
    return 0


if __name__ == "__main__":
    f = _self_test()
    initial = torch.tensor([0, 0, 0])
    final, traces = optimize_suffix(initial, steps=4, top_k=3)
    print(f"gcg_original_minimal.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
    print("final suffix:", [VOCAB[int(i)] for i in final])
    for tr in traces:
        print(tr)
