"""VLM-R1 + s1 + Safe-RLHF + 毕业 Capstone 测试."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_SRC))

from vlm_r1_minimal import (
    counting_reward, grounding_reward, format_reward_vlm, combined_vlm_reward,
)
from s1_budget_forcing import analyze_budget_distribution, budget_force_decode_mock
from safe_rlhf_minimal import LagrangianSafeRLHF, maxmin_rlhf
from capstone_graduation import MOCK_RESPONSES, GROUND_TRUTH, correctness, export_for_notebook


# ===== VLM-R1 reward =====

def test_counting_extract_answer_tag():
    assert counting_reward("<answer>5</answer>", 5) == 1.0
    assert counting_reward("<answer>3</answer>", 5) == 0.0


def test_counting_extract_inline():
    assert counting_reward("Answer: 7", 7) == 1.0
    assert counting_reward("ends with 9", 9) == 1.0


def test_grounding_perfect_iou():
    assert grounding_reward([0, 0, 100, 100], [0, 0, 100, 100]) == 1.0


def test_grounding_zero_overlap():
    assert grounding_reward([200, 200, 250, 250], [0, 0, 100, 100]) == 0.0


def test_grounding_partial_overlap_below_th():
    # half overlap → IoU = ~0.33, below 0.5 threshold → return soft value
    r = grounding_reward([0, 0, 100, 100], [50, 0, 150, 100])
    assert 0 < r < 0.5


def test_format_reward_vlm_valid():
    assert format_reward_vlm("<think>a</think><answer>b</answer>") == 1.0
    assert format_reward_vlm("plain text") == 0.0


def test_combined_keys_present():
    r = combined_vlm_reward("<think>3 cubes</think><answer>3</answer>", 3)
    assert {"format", "accuracy", "total"} == set(r.keys())
    assert r["total"] > 0.9


# ===== s1 =====

def test_budget_analyze_stats():
    responses = [
        "<think>" + " word" * 30 + "</think><answer>x</answer>",
        "<think>" + " word" * 100 + "</think><answer>x</answer>",
    ]
    stats = analyze_budget_distribution(responses)
    assert stats["count"] == 2
    assert 30 <= stats["min"] <= stats["mean"] <= stats["max"] <= 100


def test_budget_force_within_range():
    text = "<think>" + " word" * 50 + "</think><answer>x</answer>"
    out, status = budget_force_decode_mock(text, 30, 100, lambda p: "")
    assert status == "within_budget"


# ===== Safe-RLHF =====

def test_lagrangian_lambda_increases_on_violation():
    safe = LagrangianSafeRLHF(harmless_threshold=0.8, lr_lambda=0.1)
    init = safe.lam.item()
    safe.update_lambda(torch.tensor([0.1, 0.2]))   # harmless 远低于 0.8
    assert safe.lam.item() > init


def test_maxmin_keeps_min():
    r = maxmin_rlhf(torch.tensor([0.9, 0.3]), torch.tensor([0.4, 0.8]))
    assert r[0].item() == 0.4
    assert r[1].item() == 0.3


# ===== Graduation Capstone =====

def test_capstone_all_methods_present():
    assert len(MOCK_RESPONSES) == 5
    assert set(MOCK_RESPONSES.keys()) == {"1_vanilla", "2_lora", "3_adapter", "4_dpo", "5_r1_zero"}


def test_capstone_r1_has_think_format():
    r1 = MOCK_RESPONSES["5_r1_zero"]["response"]
    assert "<think>" in r1 and "</think>" in r1
    assert "<answer>" in r1
    assert "Wait" in r1 or "verify" in r1


def test_capstone_r1_correct():
    assert correctness(MOCK_RESPONSES["5_r1_zero"]["response"], GROUND_TRUTH)


def test_export_for_notebook_structure():
    out = export_for_notebook()
    assert "problem" in out and "results" in out
    assert len(out["results"]) == 5
    for r in out["results"]:
        assert {"method", "response", "correct", "response_len",
                "has_think", "has_format"} <= set(r.keys())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
