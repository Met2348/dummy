"""Capstone: DAPO 4 件套消融实验.

基座: 专题 5 capstone-A 训出的 R1-Zero baseline ckpt (mock)
增量训练: 在 baseline 上加 200 step
    config 0: 原 GRPO
    config 1: + Clip-Higher
    config 2: + Dynamic Sampling
    config 3: + Token-level PG
    config 4: + Overlong Shaping
    config 5: DAPO 全开

预期:每个 trick 各贡献 1-3pp.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_SRC = Path(__file__).parent
sys.path.insert(0, str(REPO_SRC))

from dapo_minimal import (
    asymmetric_clip_loss, is_group_useful,
    token_level_loss, response_level_loss,
    overlong_shaping,
)


def mock_ablation_run(config: dict, baseline_acc: float = 0.20) -> dict:
    """模拟每个配置的最终 accuracy + length + aha."""
    acc = baseline_acc
    length = 200.0
    if config.get("clip_higher"):
        acc += 0.025  # +2.5pp
        length += 20
    if config.get("dynamic_sampling"):
        acc += 0.015
        length += 10
    if config.get("token_level"):
        acc += 0.05    # 最大贡献
        length += 40
    if config.get("overlong"):
        acc += 0.025
        length += 15
    return {
        "accuracy": acc,
        "mean_length": length,
        "aha_freq": min(0.03 + 0.03 * sum(config.values()), 0.15),
    }


def run_ablation_grid():
    """5 个配置 + 全开 = 6 跑."""
    configs = [
        {"name": "baseline GRPO",
         "clip_higher": False, "dynamic_sampling": False,
         "token_level": False, "overlong": False},
        {"name": "+ Clip-Higher",
         "clip_higher": True, "dynamic_sampling": False,
         "token_level": False, "overlong": False},
        {"name": "+ + Dynamic Sampling",
         "clip_higher": True, "dynamic_sampling": True,
         "token_level": False, "overlong": False},
        {"name": "+ + + Token-level",
         "clip_higher": True, "dynamic_sampling": True,
         "token_level": True, "overlong": False},
        {"name": "+ + + + Overlong (DAPO full)",
         "clip_higher": True, "dynamic_sampling": True,
         "token_level": True, "overlong": True},
    ]
    print(f"{'config':35s} {'acc':6s} {'Δ':7s} {'len':7s} {'aha%':6s}")
    print("-" * 70)
    baseline = None
    for cfg in configs:
        name = cfg.pop("name")
        out = mock_ablation_run(cfg)
        delta = "—" if baseline is None else f"+{(out['accuracy']-baseline)*100:.1f}pp"
        baseline = baseline or out["accuracy"]
        print(f"{name:35s} {out['accuracy']:.1%} {delta:7s} "
              f"{out['mean_length']:.0f}     {out['aha_freq']:.1%}")


def smoke_test_dapo_components():
    print("\nDAPO 组件 smoke test:")
    # Clip-Higher
    log_p_old = torch.zeros(1, 1)
    log_p_new = torch.tensor([[0.25]])
    A = torch.tensor([1.0])
    mask = torch.ones(1, 1)
    L = asymmetric_clip_loss(log_p_new, log_p_old, A, mask, 0.2, 0.28)
    print(f"  Clip-Higher loss = {L.mean().item():.4f}")
    # Dynamic Sampling
    print(f"  Group useful [1,0,1,0] → {is_group_useful(torch.tensor([1.0, 0, 1, 0]))}")
    # Token-level vs Response-level
    per_token = torch.tensor([[1.0]*3 + [0]*2, [1.0]*5])
    mask = torch.tensor([[1.0]*3 + [0]*2, [1.0]*5])
    L_t = token_level_loss(per_token, mask)
    L_r = response_level_loss(per_token, mask)
    print(f"  Token-level = {L_t:.4f}  Response-level = {L_r:.4f}")
    # Overlong shaping
    out = overlong_shaping(torch.tensor([1.0]), torch.tensor([6000.0]), 4096, 200)
    print(f"  Overlong 6000/4096 reward = {out.item():.4f}")


if __name__ == "__main__":
    print("Capstone — DAPO 4 件套消融实验 (mock)\n" + "=" * 70)
    run_ablation_grid()
    smoke_test_dapo_components()
    print("\n→ 实际训练: 每 config 200 step on baseline ckpt, 5090 24GB 每跑 1h.")
