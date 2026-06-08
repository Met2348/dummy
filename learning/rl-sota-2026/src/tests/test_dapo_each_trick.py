"""DAPO 4 件套独立单元测试."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_SRC))

from dapo_minimal import (
    asymmetric_clip_loss, is_group_useful,
    token_level_loss, response_level_loss,
    overlong_shaping,
)


# ===== Clip-Higher =====

def test_clip_higher_more_permissive_than_sym():
    """A > 0 时，Clip-Higher 允许更大的 ratio → 损失更小（更好）."""
    log_p_old = torch.zeros(1, 1)
    log_p_new = torch.tensor([[0.25]])     # ratio ≈ 1.28
    A = torch.tensor([1.0])
    mask = torch.ones(1, 1)
    L_sym = asymmetric_clip_loss(log_p_new, log_p_old, A, mask,
                                  eps_low=0.2, eps_high=0.2).item()
    L_high = asymmetric_clip_loss(log_p_new, log_p_old, A, mask,
                                   eps_low=0.2, eps_high=0.28).item()
    # sym clip = 1.2 → -min(1.28, 1.2) = -1.2 → loss = 1.2 → 但 surr1 = 1.28 → min(1.28, 1.2) = 1.2
    # high clip = 1.28 → min(1.28, 1.28) = 1.28
    # 损失 = -min(surr1, surr2) → sym 损失 = -1.2; high 损失 = -1.28
    # 即 L_high < L_sym (更负 = 更小)
    assert L_high < L_sym, (L_sym, L_high)


# ===== Dynamic Sampling =====

def test_group_useful_all_pass():
    """全对 → 不可用."""
    assert not is_group_useful(torch.tensor([1.0, 1, 1]))


def test_group_useful_all_fail():
    """全错 → 不可用."""
    assert not is_group_useful(torch.tensor([0.0, 0, 0]))


def test_group_useful_mixed():
    """有对有错 → 可用."""
    assert is_group_useful(torch.tensor([1.0, 0, 1, 0]))


def test_group_useful_pm_one_rewards():
    """论文使用 +1/-1 reward，混合正确/错误也应可用."""
    assert is_group_useful(torch.tensor([1.0, -1, 1, -1]))


# ===== Token-level vs Response-level =====

def test_token_level_long_response_more_weight():
    """长 response 的 per-token 在 token-level 中权重等于短 response 的 per-token."""
    per_token = torch.tensor([
        [2.0, 2.0, 0.0, 0.0],   # response len 2
        [1.0, 1.0, 1.0, 1.0],   # response len 4
    ])
    mask = torch.tensor([
        [1, 1, 0, 0],
        [1, 1, 1, 1],
    ], dtype=torch.float32)
    L_token = token_level_loss(per_token, mask)
    L_resp = response_level_loss(per_token, mask)
    # token-level: (2+2+1+1+1+1)/6 = 8/6 = 1.33
    # response-level: ((4/2) + (4/4))/2 = (2 + 1)/2 = 1.5
    assert abs(L_token - 8 / 6) < 1e-4
    assert abs(L_resp - 1.5) < 1e-4
    assert L_token != L_resp


# ===== Overlong Shaping =====

def test_overlong_unmodified_short():
    """response 短于 expected_len → reward 不变."""
    out = overlong_shaping(
        torch.tensor([1.0]), torch.tensor([1000.0]),
        expected_len=4096,
    )
    assert abs(out.item() - 1.0) < 1e-6


def test_overlong_penalized_long():
    """response 远长于 max_len → 正确样本被加 -1 penalty."""
    out = overlong_shaping(
        torch.tensor([1.0]), torch.tensor([10000.0]),
        expected_len=4096, cache_len=200,
    )
    assert abs(out.item() - 0.0) < 1e-6


def test_overlong_linear_cache_penalty():
    """cache 区间内线性扣分."""
    out = overlong_shaping(
        torch.tensor([1.0]), torch.tensor([4196.0]),
        expected_len=4096, cache_len=200,
    )
    assert abs(out.item() - 0.5) < 1e-6


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
