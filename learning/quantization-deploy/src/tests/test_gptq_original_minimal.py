"""Tests for the paper-shaped GPTQ toy implementation."""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gptq_original_minimal import (
    calibration_hessian,
    gptq_columnwise,
    make_symmetric_row_grid,
    naive_round,
    reconstruction_mse,
    toy_comparison,
)


def test_grid_has_one_scale_per_output_row():
    W = torch.tensor([[1.0, -2.0, 0.5], [0.25, -0.5, 0.75]])
    grid = make_symmetric_row_grid(W, n_bits=4)
    assert grid.scale.shape == (2,)
    assert grid.qmax == 7


def test_hessian_shape_and_positive_diagonal():
    torch.manual_seed(0)
    X = torch.randn(32, 12)
    H = calibration_hessian(X)
    assert H.shape == (12, 12)
    assert torch.all(H.diag() > 0)


def test_gptq_columnwise_shapes_and_finite_values():
    torch.manual_seed(0)
    W = torch.randn(10, 20)
    X = torch.randn(64, 20)
    Q, E = gptq_columnwise(W, X, n_bits=4)
    assert Q.shape == W.shape
    assert E.shape == W.shape
    assert torch.isfinite(Q).all()
    assert torch.isfinite(E).all()


def test_toy_comparison_has_reasonable_reconstruction_error():
    torch.manual_seed(0)
    W = torch.randn(24, 48)
    X = torch.randn(192, 48)
    Q, _ = gptq_columnwise(W, X, n_bits=4)
    naive = naive_round(W, n_bits=4)
    assert reconstruction_mse(W, Q, X) <= reconstruction_mse(W, naive, X) * 1.7


def test_seed_zero_demo_shows_compensation_can_help():
    result = toy_comparison(seed=0)
    assert result["gptq_mse"] <= result["naive_mse"] * 1.05


if __name__ == "__main__":
    test_grid_has_one_scale_per_output_row()
    test_hessian_shape_and_positive_diagonal()
    test_gptq_columnwise_shapes_and_finite_values()
    test_toy_comparison_has_reasonable_reconstruction_error()
    test_seed_zero_demo_shows_compensation_can_help()
    print("test_gptq_original_minimal passed")
