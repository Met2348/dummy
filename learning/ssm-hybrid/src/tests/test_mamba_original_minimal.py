"""Tests for the original Mamba paper mechanisms."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mamba_original_minimal import (  # noqa: E402
    recurrent_loop,
    selective_copy_toy,
    selective_ssm_reference,
    sequential_affine_prefix_scan,
    tiny_s6_inputs,
)


def test_selective_gate_keeps_last_marked_value():
    _, selective, fixed = selective_copy_toy()

    expected = torch.tensor([5.0, 5.0, 5.0, 9.0, 9.0, 9.0])
    assert torch.allclose(selective, expected)

    # A fixed gate has no content-aware write/ignore decision, so zeros blur memory.
    assert not torch.allclose(fixed, expected)
    assert fixed[2] < selective[2]


def test_affine_prefix_scan_matches_recurrent_loop():
    a = torch.tensor([0.5, 0.25, 0.8, 0.1])
    b = torch.tensor([1.0, -0.5, 2.0, 0.25])

    _, prefix_b = sequential_affine_prefix_scan(a, b)
    states = recurrent_loop(a, b)

    assert torch.allclose(prefix_b, states)


def test_selective_ssm_reference_shape_and_finiteness():
    u, delta, A, B, C = tiny_s6_inputs()
    y = selective_ssm_reference(u, delta, A, B, C)

    assert y.shape == u.shape
    assert torch.isfinite(y).all()


def test_delta_changes_how_fast_state_forgets():
    u, delta, A, B, C = tiny_s6_inputs()

    slow_delta = delta * 0.05
    fast_delta = delta * 4.0
    y_slow = selective_ssm_reference(u, slow_delta, A, B, C)
    y_fast = selective_ssm_reference(u, fast_delta, A, B, C)

    assert not torch.allclose(y_slow, y_fast)
    assert torch.isfinite(y_slow).all()
    assert torch.isfinite(y_fast).all()
