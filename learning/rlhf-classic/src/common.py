"""RLHF Classic 通用工具。

继承专题 1（GAE / log_prob / KL）并新增 RLHF 专用：
- Anthropic-HH 数据 loader
- preference pair → input format
- Bradley-Terry log-likelihood helper
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_RL_SRC = Path(__file__).parent.parent.parent / "rl-foundations" / "src"
if REPO_RL_SRC.exists():
    sys.path.insert(0, str(REPO_RL_SRC))
    # 复用 rl-foundations 的 common
    from common import (        # noqa: F401
        compute_gae, normalize_advantages, categorical_log_prob,
        categorical_entropy, kl_categorical, kl_approx_logp,
        explained_variance, RolloutBuffer, set_seed,
    )


def load_hh_pairs(n: int = 1000, split: str = "train"):
    """加载 Anthropic-HH chosen/rejected pair。"""
    from datasets import load_dataset
    ds = load_dataset("Anthropic/hh-rlhf", split=f"{split}[:{n}]")
    return ds


def bt_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> torch.Tensor:
    """Bradley-Terry preference loss.

    L = -E[ log sigmoid(r_chosen - r_rejected) ]
    """
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


def pad_and_mask(
    sequences: list[list[int]],
    pad_token_id: int,
    max_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """list of token id list → padded tensor + attention_mask."""
    if max_len is None:
        max_len = max(len(s) for s in sequences)
    out = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for i, s in enumerate(sequences):
        L = min(len(s), max_len)
        out[i, :L] = torch.tensor(s[:L])
        mask[i, :L] = 1
    return out, mask
