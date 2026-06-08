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
from vision_r1_original_minimal import (
    MultimodalQA,
    PTSTStage,
    build_cold_start_sample,
    group_relative_advantages,
    grpo_clipped_surrogate_terms,
    hard_format_result_reward,
    score_stage_group,
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
    # Half overlap means IoU is about 0.33, below the threshold.
    r = grounding_reward([0, 0, 100, 100], [50, 0, 150, 100])
    assert 0 < r < 0.5


def test_format_reward_vlm_valid():
    assert format_reward_vlm("<think>a</think><answer>b</answer>") == 1.0
    assert format_reward_vlm("plain text") == 0.0


def test_combined_keys_present():
    r = combined_vlm_reward("<think>3 cubes</think><answer>3</answer>", 3)
    assert {"format", "accuracy", "total"} == set(r.keys())
    assert r["total"] > 0.9


# ===== Vision-R1 paper mechanisms =====

def test_vision_r1_cold_start_bridging_keeps_visual_facts():
    item = MultimodalQA(
        image_facts=("AF is 10 units.", "AD is 3.5 units.", "BD option is A."),
        question="Find BD. Options: A 3, B 3.5, C 6, D 7.",
        answer="A",
    )
    sample = build_cold_start_sample(item)
    assert "AF is 10 units" in sample.detailed_description
    assert "BD option is A" in sample.detailed_description
    assert hard_format_result_reward(sample.response, "A") == 1.0


def test_hard_format_result_reward_requires_both_conditions():
    good = "<think>check geometry</think><answer>Final Answer:C</answer>"
    bad_format = "Final Answer:C"
    wrong = "<think>check geometry</think><answer>Final Answer:D</answer>"
    assert hard_format_result_reward(good, "C") == 1.0
    assert hard_format_result_reward(bad_format, "C") == 0.0
    assert hard_format_result_reward(wrong, "C") == 0.0


def test_group_advantage_and_grpo_terms_are_shaped():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    adv = group_relative_advantages(rewards)
    assert adv.shape == rewards.shape
    assert abs(adv.mean().item()) < 1e-6

    terms = grpo_clipped_surrogate_terms(
        logp_new=torch.tensor([-1.0, -1.2, -0.8, -1.4]),
        logp_old=torch.tensor([-1.1, -1.1, -0.9, -1.3]),
        logp_ref=torch.tensor([-1.0, -1.0, -1.0, -1.0]),
        rewards=rewards,
    )
    assert {"advantages", "ratio", "kl", "objective_terms"} == set(terms)
    assert terms["objective_terms"].shape == rewards.shape


def test_ptst_stage_filters_length_before_rewarding():
    short_good = "<think>short correct</think><answer>Final Answer:A</answer>"
    long_good = "<think>" + " token" * 12 + "</think><answer>Final Answer:A</answer>"
    wrong = "<think>short wrong</think><answer>Final Answer:B</answer>"
    stage = PTSTStage("toy", max_tokens=5, group_size=8, steps=1)
    rewards = score_stage_group([short_good, long_good, wrong], "A", stage)
    assert rewards.tolist() == [1.0, 0.0]


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
    assert abs(r[0].item() - 0.4) < 1e-6
    assert abs(r[1].item() - 0.3) < 1e-6


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
