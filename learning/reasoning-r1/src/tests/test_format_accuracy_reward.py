"""R1 reward 函数严格单元测试."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_SRC))

from rewards.format_reward import format_reward, extract_answer
from rewards.accuracy_reward import (
    gsm8k_extract_answer, gsm8k_reward, countdown_reward
)


# ===== format_reward =====

def test_format_correct():
    s = "<think>step 1</think><answer>42</answer>"
    assert format_reward(s) == 1.0


def test_format_with_whitespace():
    s = "  <think>step</think>\n<answer>42</answer>  "
    assert format_reward(s) == 1.0


def test_format_missing_think():
    assert format_reward("<answer>42</answer>") == 0.0


def test_format_missing_answer():
    assert format_reward("<think>step</think>") == 0.0


def test_format_empty_think():
    assert format_reward("<think></think><answer>42</answer>") == 0.0


def test_format_text_before():
    """<think> 前有其它字符应失败（严格模式）。"""
    s = "Let me think. <think>...</think><answer>42</answer>"
    assert format_reward(s) == 0.0


def test_extract_answer_simple():
    assert extract_answer("<answer>42</answer>") == "42"


def test_extract_answer_multiline():
    assert extract_answer("<answer>\n  42\n</answer>") == "42"


# ===== gsm8k_extract_answer =====

def test_gsm8k_extract():
    assert gsm8k_extract_answer("the answer is 42. #### 42") == "42"


def test_gsm8k_extract_with_commas():
    assert gsm8k_extract_answer("…#### 1,234") == "1234"


def test_gsm8k_extract_negative():
    assert gsm8k_extract_answer("#### -5") == "-5"


def test_gsm8k_extract_missing():
    assert gsm8k_extract_answer("answer: 42") is None


# ===== gsm8k_reward =====

def test_gsm8k_reward_exact():
    assert gsm8k_reward("42", "42") == 1.0


def test_gsm8k_reward_float_equiv():
    assert gsm8k_reward("42.0", "42") == 1.0


def test_gsm8k_reward_wrong():
    assert gsm8k_reward("41", "42") == 0.0


def test_gsm8k_reward_none():
    assert gsm8k_reward(None, "42") == 0.0


# ===== countdown_reward =====

def test_countdown_correct():
    assert countdown_reward("3*4+5", [3, 4, 5], 17) == 1.0


def test_countdown_correct_with_parens():
    assert countdown_reward("(3+4)*5", [3, 4, 5], 35) == 1.0


def test_countdown_wrong_result():
    assert countdown_reward("3+4+5", [3, 4, 5], 17) == 0.0


def test_countdown_wrong_numbers():
    """用了不在 numbers 列表中的数。"""
    assert countdown_reward("3+4+5+6", [3, 4, 5], 18) == 0.0


def test_countdown_none():
    assert countdown_reward(None, [3, 4, 5], 17) == 0.0


def test_countdown_invalid_expr():
    assert countdown_reward("3 oops 4", [3, 4, 5], 17) == 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
