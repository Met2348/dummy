"""PRM + BoN + RLVR + MCTS 单元测试."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_SRC))

from prm_minimal import aggregate_step_scores, PRMHead
from bon_search import best_of_n, majority_vote, weighted_bon, bon_compare_strategies
from rlvr_demo import gsm8k_reward, format_reward, equation_reward
from prime_minimal import implicit_prm_per_token, aggregate_to_step
from math_shepherd_data_gen import label_from_rollouts


# ===== PRM aggregation =====

def test_aggregate_modes_differ():
    logits = torch.tensor([
        [3.0, 0, 0],   # good=high
        [3.0, 0, 0],
        [0, 0, 3.0],   # bad=high
    ])
    s_mean = aggregate_step_scores(logits, "mean")
    s_min = aggregate_step_scores(logits, "min")
    s_ml = aggregate_step_scores(logits, "min_last")
    s_prod = aggregate_step_scores(logits, "product")
    assert s_min < s_mean
    assert s_ml < s_mean


def test_prm_head_shape():
    head = PRMHead(hidden_size=32, num_labels=3)
    h = torch.randn(4, 32)
    out = head(h)
    assert out.shape == (4, 3)


# ===== BoN =====

def test_bon_picks_max():
    ans, idx, score = best_of_n(["a", "b", "c"], [0.1, 0.9, 0.4])
    assert ans == "b" and idx == 1


def test_majority_basic():
    m, c = majority_vote(["x", "y", "x", "x", "y"])
    assert m == "x"


def test_weighted_bon_aggregates_scores():
    answers = ["A", "B", "A"]
    scores = [0.3, 0.9, 0.4]
    m, w = weighted_bon(answers, scores)
    # A: 0.7, B: 0.9 → B 应胜
    assert m == "B"


def test_compare_all_strategies_runs():
    out = bon_compare_strategies(["7", "7", "8"], [0.9, 0.6, 0.4], "7")
    assert out["greedy"][1]
    assert out["bon"][1]


# ===== RLVR =====

def test_gsm8k_extracts_after_hash():
    assert gsm8k_reward("blah #### 42", "42") == 1.0
    assert gsm8k_reward("blah #### 0", "42") == 0.0
    assert gsm8k_reward("no hash here", "42") == 0.0


def test_format_reward_full_match():
    assert format_reward("<think>x</think><answer>y</answer>") == 1.0
    assert format_reward("only <think>x</think>") == 0.0


def test_equation_reward_countdown():
    assert equation_reward("6*4 = 24", 24) == 1.0
    assert equation_reward("(8-2)*4 = 24", 24) == 1.0
    assert equation_reward("2+3 = 5", 24) == 0.0


# ===== PRIME =====

def test_implicit_prm_shape():
    log_p_a = torch.randn(2, 10)
    log_p_r = torch.randn(2, 10)
    r = implicit_prm_per_token(log_p_a, log_p_r, beta=0.05)
    assert r.shape == (2, 10)


def test_prime_aggregate_to_step_count():
    per_token = torch.arange(12, dtype=torch.float32).expand(1, 12)
    step_end = [2, 5, 11]
    sr = aggregate_to_step(per_token, step_end)
    assert sr.shape == (1, 3)


# ===== Math-Shepherd =====

def test_label_from_rollouts_boundary():
    assert label_from_rollouts(8, 10) == "good"
    assert label_from_rollouts(2, 10) == "bad"
    assert label_from_rollouts(5, 10) == "neutral"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
