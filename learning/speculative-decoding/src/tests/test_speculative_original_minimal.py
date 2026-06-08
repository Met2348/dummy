"""Tests for the paper-shaped speculative decoding examples."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from speculative_original_minimal import (
    best_gamma,
    exact_one_step_output_distribution,
    expected_tokens_per_iteration,
    overlap_mass,
    residual_distribution,
    summarize,
    walltime_speedup,
)


def test_exact_one_step_distribution_equals_target():
    p = [0.50, 0.30, 0.20]
    q = [0.35, 0.45, 0.20]
    out = exact_one_step_output_distribution(p, q)
    assert all(abs(a - b) < 1e-9 for a, b in zip(p, out))


def test_residual_distribution_uses_only_target_surplus():
    p = [0.50, 0.30, 0.20]
    q = [0.35, 0.45, 0.20]
    residual = residual_distribution(p, q)
    assert residual == [1.0, 0.0, 0.0]


def test_overlap_mass_is_acceptance_rate():
    p = [0.50, 0.30, 0.20]
    q = [0.35, 0.45, 0.20]
    assert abs(overlap_mass(p, q) - 0.85) < 1e-9


def test_expected_tokens_matches_geometric_sum():
    value = expected_tokens_per_iteration(alpha=0.75, gamma=3)
    assert abs(value - (1.0 + 0.75 + 0.75**2 + 0.75**3)) < 1e-9


def test_speedup_penalizes_expensive_draft_model():
    cheap = walltime_speedup(alpha=0.75, gamma=7, draft_cost_ratio=0.02)
    expensive = walltime_speedup(alpha=0.75, gamma=7, draft_cost_ratio=0.20)
    assert cheap > expensive


def test_best_gamma_returns_positive_speedup():
    gamma, speed = best_gamma(alpha=0.75, draft_cost_ratio=0.02, max_gamma=16)
    assert gamma >= 1
    assert speed > 1.0


def test_summarize_contains_paper_case():
    result = summarize()
    assert result["t5_small_expected_tokens"] > 3.0
    assert result["t5_small_speedup_model"] > 2.0


if __name__ == "__main__":
    test_exact_one_step_distribution_equals_target()
    test_residual_distribution_uses_only_target_surplus()
    test_overlap_mass_is_acceptance_rate()
    test_expected_tokens_matches_geometric_sum()
    test_speedup_penalizes_expensive_draft_model()
    test_best_gamma_returns_positive_speedup()
    test_summarize_contains_paper_case()
    print("test_speculative_original_minimal passed")
