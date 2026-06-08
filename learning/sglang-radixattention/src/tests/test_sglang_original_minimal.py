"""Tests for the paper-shaped SGLang examples."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sglang_original_minimal import (
    forced_literal_jump,
    lm_program_prompts,
    measure_radix_reuse,
    summarize,
)


def test_program_prompts_have_shared_prefixes():
    prompts = lm_program_prompts()
    assert len(prompts) >= 10
    assert prompts[0][:10] == prompts[1][:10]


def test_radix_reuse_saves_prefill_tokens():
    report = measure_radix_reuse(lm_program_prompts())
    assert report.saved_prefill_tokens > 0
    assert report.hit_rate > 0.50
    assert report.radix_prefill_tokens < report.naive_prefill_tokens


def test_forced_literal_jump_skips_multiple_steps():
    jump = forced_literal_jump()
    assert jump["forced"] == jump["literal"]
    assert jump["normal_decode_steps"] > jump["jump_forward_steps"]


def test_summarize_has_required_metrics():
    result = summarize()
    for key in (
        "naive_prefill_tokens",
        "radix_prefill_tokens",
        "saved_prefill_tokens",
        "hit_rate",
        "jump_forward",
    ):
        assert key in result


if __name__ == "__main__":
    test_program_prompts_have_shared_prefixes()
    test_radix_reuse_saves_prefill_tokens()
    test_forced_literal_jump_skips_multiple_steps()
    test_summarize_has_required_metrics()
    print("test_sglang_original_minimal passed")
