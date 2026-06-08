"""Graduation capstone 测试."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from vanilla_gpt2 import GPT2Config, VanillaGPT2
from train_variant import VARIANTS
from bench_matrix import EXPECTED, METRICS, ablation_breakdown
from common import CKPT_VARIANTS, variant_desc
from tinystories_original_minimal import (
    ModelSpec,
    StoryConstraints,
    build_story_prompt,
    capability_profile,
    child_vocab_fraction,
    gpt_eval_prompt,
    lexical_diversity,
    nearest_training_overlap,
    next_token_cross_entropy,
    rouge_k_fmeasure,
    rouge_k_precision,
    rough_gpt_params,
    toy_teacher_scores,
)


def test_vanilla_gpt2_forward():
    c = GPT2Config(vocab_size=100, hidden=64, n_head=4,
                    n_layer=2, seq_len=64)
    m = VanillaGPT2(c)
    x = torch.randint(0, 100, (2, 32))
    out = m(x)
    assert out.shape == (2, 32, 100)


def test_vanilla_gpt2_params():
    c = GPT2Config()
    m = VanillaGPT2(c)
    n = sum(p.numel() for p in m.parameters())
    n_no_embed = n - m.tok_embed.weight.numel()
    assert 80e6 < n_no_embed < 100e6


def test_all_variants_configured():
    for v in CKPT_VARIANTS:
        assert v in VARIANTS
        cfg = VARIANTS[v]
        assert cfg.max_step > 0
        assert cfg.base_lr > 0


def test_variant_progression():
    """E should have lowest val_loss; A highest."""
    losses = {v: EXPECTED[v]["val_loss"] for v in CKPT_VARIANTS}
    assert losses["A"] > losses["B"] > losses["C"] >= losses["E"]


def test_niah_only_d_e_high():
    assert EXPECTED["A"]["niah_8k"] == 0
    assert EXPECTED["B"]["niah_8k"] == 0
    assert EXPECTED["D"]["niah_8k"] > 0.5
    assert EXPECTED["E"]["niah_8k"] > 0.5


def test_ablation_data_contributes_hellaswag():
    abl = ablation_breakdown(EXPECTED)
    assert abl["data (A->B)"]["hellaswag"] >= 0.04


def test_ablation_long_ctx_only_niah():
    abl = ablation_breakdown(EXPECTED)
    diff = abl["long_ctx (C->D)"]
    assert diff["niah_8k"] >= 0.5
    assert abs(diff["hellaswag"]) < 0.05


def test_variant_descriptions():
    for v in CKPT_VARIANTS:
        assert len(variant_desc(v)) > 5


def test_tinystories_prompt_controls_words_and_features():
    prompt = build_story_prompt(
        StoryConstraints(
            verb="decorate",
            noun="thunder",
            adjective="ancient",
            features=("dialogue", "bad ending"),
        )
    )
    assert "decorate" in prompt
    assert "thunder" in prompt
    assert "ancient" in prompt
    assert "dialogue" in prompt
    assert "simple words" in prompt


def test_child_vocab_fraction_rewards_simple_distribution():
    simple = "the happy dog went to the tree"
    hard = "quantum governance optimized semiconductor portfolios"
    assert child_vocab_fraction(simple) > child_vocab_fraction(hard)
    assert child_vocab_fraction(simple) == 1.0


def test_rouge_overlap_detects_copying():
    story = "Lily saw a red ball and helped a sad dog"
    copied = "Lily saw a red ball and helped a sad dog"
    changed = "Tom baked bread with mom in the house"
    assert rouge_k_precision(copied, story, k=2) == 1.0
    assert rouge_k_fmeasure(copied, story, k=2) == 1.0
    assert rouge_k_precision(changed, story, k=2) < 0.2
    assert nearest_training_overlap(copied, [story, changed], k=2) == 1.0


def test_toy_gpt_eval_contract_has_rubric_dimensions():
    beginning = "Once upon a time, Lily had a red ball."
    completion = "Lily played with the ball. She was happy."
    prompt = gpt_eval_prompt(beginning, completion)
    scores = toy_teacher_scores(beginning, completion)
    assert "grammar" in prompt and "creativity" in prompt
    assert {"grammar", "creativity", "consistency", "plot"} == set(scores)
    assert scores["consistency"] >= 5.0


def test_width_depth_profile_and_loss_shape():
    shallow = ModelSpec(hidden=256, layers=1, heads=4)
    deep = ModelSpec(hidden=256, layers=8, heads=4)
    wide = ModelSpec(hidden=1024, layers=1, heads=8)
    assert rough_gpt_params(deep) > rough_gpt_params(shallow)
    assert rough_gpt_params(wide) > rough_gpt_params(shallow)
    assert capability_profile(deep)["context_tracking"] > (
        capability_profile(shallow)["context_tracking"]
    )
    assert capability_profile(wide)["factual_knowledge"] > (
        capability_profile(shallow)["factual_knowledge"]
    )

    logits = torch.randn(2, 4, 10)
    targets = torch.randint(0, 10, (2, 4))
    loss = next_token_cross_entropy(logits, targets)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_lexical_diversity_positive_for_multiple_stories():
    stories = [
        "the dog and the cat play",
        "the girl and the boy run",
    ]
    assert 0 < lexical_diversity(stories) < 1
