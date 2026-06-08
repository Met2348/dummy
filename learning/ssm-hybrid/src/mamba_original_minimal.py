"""Minimal mechanisms from the Mamba paper.

This file is intentionally small and CPU-friendly. It does not try to
reproduce the fused CUDA kernel. Instead it exposes the paper's core ideas:

1. LTI SSMs use fixed dynamics across positions.
2. Selective SSMs make Delta, B, and C depend on the current token.
3. The recurrence can be viewed as a scan over affine maps.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AffineMap:
    """A scalar map h_out = a * h_in + b."""

    a: torch.Tensor
    b: torch.Tensor


def compose_affine(left: AffineMap, right: AffineMap) -> AffineMap:
    """Return right(left(h)).

    If left(h) = a1 * h + b1 and right(h) = a2 * h + b2, then
    right(left(h)) = (a2 * a1) * h + (b2 + a2 * b1).
    This associativity is the scan handle used by selective recurrence.
    """

    return AffineMap(a=right.a * left.a, b=right.b + right.a * left.b)


def sequential_affine_prefix_scan(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute prefix affine maps for h_t = a_t * h_{t-1} + b_t.

    Args:
        a: Tensor with shape (T,).
        b: Tensor with shape (T,).

    Returns:
        Two tensors with shape (T,), the prefix a and b. With h_0 = 0,
        the hidden state after each prefix is simply prefix_b.
    """

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b must be rank-1 tensors")
    if a.shape != b.shape:
        raise ValueError("a and b must have identical shape")

    prefix_a = []
    prefix_b = []
    current = AffineMap(
        a=torch.ones((), dtype=a.dtype, device=a.device),
        b=torch.zeros((), dtype=b.dtype, device=b.device),
    )
    for a_t, b_t in zip(a, b):
        current = compose_affine(current, AffineMap(a=a_t, b=b_t))
        prefix_a.append(current.a)
        prefix_b.append(current.b)
    return torch.stack(prefix_a), torch.stack(prefix_b)


def recurrent_loop(a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
    """Reference loop for h_t = a_t * h_{t-1} + b_t."""

    if h0 is None:
        h = torch.zeros((), dtype=b.dtype, device=b.device)
    else:
        h = h0
    states = []
    for a_t, b_t in zip(a, b):
        h = a_t * h + b_t
        states.append(h)
    return torch.stack(states)


def gated_selective_memory(values: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Toy version of Theorem 1 in the paper.

    h_t = (1 - gate_t) * h_{t-1} + gate_t * value_t

    gate_t near 1 writes the current value. gate_t near 0 keeps memory.
    This is the simplest way to see why input-dependent Delta acts like
    a content-aware write/ignore decision.
    """

    if values.ndim != 1 or gate.ndim != 1:
        raise ValueError("values and gate must be rank-1 tensors")
    if values.shape != gate.shape:
        raise ValueError("values and gate must have identical shape")

    h = torch.zeros((), dtype=values.dtype, device=values.device)
    out = []
    for x_t, g_t in zip(values, gate):
        h = (1.0 - g_t) * h + g_t * x_t
        out.append(h)
    return torch.stack(out)


def fixed_gate_memory(values: torch.Tensor, gate_value: float) -> torch.Tensor:
    """LTI-style contrast: the write gate is fixed at every position."""

    gate = torch.full_like(values, float(gate_value))
    return gated_selective_memory(values, gate)


def selective_copy_toy() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return values, selective output, and fixed-gate output for a toy task."""

    values = torch.tensor([5.0, 0.0, 0.0, 9.0, 0.0, 0.0])
    colored_token_gate = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    selective = gated_selective_memory(values, colored_token_gate)
    fixed = fixed_gate_memory(values, gate_value=0.35)
    return values, selective, fixed


def discretize_zoh_diagonal(delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-order hold discretization for diagonal A.

    Args:
        delta: Shape (T,), positive step sizes.
        A: Shape (N,), negative diagonal state matrix.
        B: Shape (T, N), input-dependent input projection.

    Returns:
        A_bar and B_bar, each with shape (T, N).
    """

    if delta.ndim != 1 or A.ndim != 1 or B.ndim != 2:
        raise ValueError("expected delta (T,), A (N,), B (T, N)")
    if B.shape != (delta.shape[0], A.shape[0]):
        raise ValueError("B must have shape (T, N)")

    dA = delta[:, None] * A[None, :]
    A_bar = torch.exp(dA)
    B_bar = (A_bar - 1.0) / A[None, :] * B
    return A_bar, B_bar


def selective_ssm_reference(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    """Reference S6 recurrence with input-dependent delta, B, and C.

    Args:
        u: Shape (T,), one input channel.
        delta: Shape (T,), input-dependent positive step sizes.
        A: Shape (N,), negative diagonal state matrix.
        B: Shape (T, N), input-dependent write projection.
        C: Shape (T, N), input-dependent read projection.

    Returns:
        y: Shape (T,), one output channel.
    """

    if u.ndim != 1 or delta.ndim != 1 or A.ndim != 1 or B.ndim != 2 or C.ndim != 2:
        raise ValueError("expected u (T,), delta (T,), A (N,), B (T, N), C (T, N)")
    if u.shape != delta.shape:
        raise ValueError("u and delta must have identical shape")
    if B.shape != C.shape or B.shape != (u.shape[0], A.shape[0]):
        raise ValueError("B and C must have shape (T, N)")

    A_bar, B_bar = discretize_zoh_diagonal(delta, A, B)
    h = torch.zeros_like(A)
    out = []
    for t in range(u.shape[0]):
        h = A_bar[t] * h + B_bar[t] * u[t]
        out.append(torch.dot(C[t], h))
    return torch.stack(out)


def tiny_s6_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct a deterministic tiny selective SSM example."""

    u = torch.tensor([1.0, 0.0, 2.0, 0.0, 3.0])
    delta = torch.tensor([1.0, 0.05, 1.0, 0.05, 1.0])
    A = torch.tensor([-1.0, -2.0])
    B = torch.tensor(
        [
            [1.0, 0.2],
            [0.0, 0.0],
            [0.3, 1.0],
            [0.0, 0.0],
            [1.0, 0.5],
        ]
    )
    C = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )
    return u, delta, A, B, C


if __name__ == "__main__":
    values, selective, fixed = selective_copy_toy()
    print("values:", values.tolist())
    print("selective memory:", selective.tolist())
    print("fixed-gate memory:", [round(x, 3) for x in fixed.tolist()])

    u, delta, A, B, C = tiny_s6_inputs()
    print("tiny S6 output:", [round(x, 4) for x in selective_ssm_reference(u, delta, A, B, C).tolist()])
