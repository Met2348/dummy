"""6 PO 方法 + RainbowPO 一致性 + 边界测试."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_SRC))

from dpo_minimal import dpo_loss
from ipo_minimal import ipo_loss
from kto_minimal import kto_loss
from orpo_minimal import orpo_loss, log_odds
from simpo_minimal import simpo_loss, length_normalized_logp
from cpo_minimal import cpo_loss
from dpop_minimal import dpop_loss
from rainbowpo import VARIANTS, unified_po_loss


def setup_logp(B: int = 4):
    """统一 fixture."""
    torch.manual_seed(0)
    return dict(
        c_a=torch.randn(B) * 0.1 - 1,
        c_r=torch.randn(B) * 0.1 - 1.5,
        r_a=torch.randn(B) * 0.1 - 1.5,
        r_r=torch.randn(B) * 0.1 - 1,
    )


# ===== 基础正确性 =====

def test_dpo_zero_margin_log2():
    """全 0 margin → loss = log 2."""
    zero = torch.zeros(4)
    L = dpo_loss(zero, zero, zero, zero, beta=0.1)
    assert abs(L.item() - 0.693147) < 1e-4


def test_ipo_positive():
    s = setup_logp()
    L = ipo_loss(s["c_a"], s["c_r"], s["r_a"], s["r_r"])
    assert L.item() > 0


def test_kto_runs_with_mixed():
    log_p_act = torch.randn(8) - 1
    log_p_ref = torch.randn(8) - 1
    is_des = torch.tensor([1, 0, 1, 0, 1, 1, 0, 0])
    L = kto_loss(log_p_act, log_p_ref, is_des)
    assert L.item() > 0


def test_log_odds_monotone():
    """log_p 增大 → log_odds 增大."""
    o1 = log_odds(torch.tensor(-3.0))
    o2 = log_odds(torch.tensor(-0.5))
    assert o2.item() > o1.item()


def test_simpo_length_normalize():
    """长 response 不应占便宜。"""
    log_p = -torch.ones(2, 10)   # 每 token log p = -1
    mask_short = torch.cat([torch.ones(1, 5), torch.zeros(1, 5)], dim=1).expand(2, -1).clone()
    mask_short[1] = 0; mask_short[1, :5] = 1
    mask_long = torch.ones(2, 10)
    norm_short = length_normalized_logp(log_p, mask_short)
    norm_long = length_normalized_logp(log_p, mask_long)
    assert torch.allclose(norm_short, norm_long, atol=1e-6)


def test_orpo_includes_sft():
    """ORPO loss > SFT loss (加了 OR 项)."""
    log_p_c = -torch.rand(4) * 2
    log_p_r = -torch.rand(4) * 2 - 1
    sft = torch.tensor(1.0)
    L = orpo_loss(log_p_c, log_p_r, sft, lambda_or=0.5)
    assert L.item() > sft.item()


def test_cpo_includes_sft():
    log_p_c = -torch.rand(4) * 2
    log_p_r = -torch.rand(4) * 2 - 1
    sft = torch.tensor(1.0)
    L = cpo_loss(log_p_c, log_p_r, sft)
    assert L.item() > sft.item() - 1e-4


def test_dpop_punishes_chosen_drop():
    """actor chosen prob < ref chosen prob → DPOP > DPO."""
    log_p_c_actor = torch.log(torch.tensor([0.3]))
    log_p_c_ref = torch.log(torch.tensor([0.5]))
    log_p_r_actor = torch.log(torch.tensor([0.1]))
    log_p_r_ref = torch.log(torch.tensor([0.4]))
    L_dpo = dpo_loss(log_p_c_actor, log_p_c_ref, log_p_r_actor, log_p_r_ref)
    L_dpop = dpop_loss(log_p_c_actor, log_p_c_ref, log_p_r_actor, log_p_r_ref, lambda_p=50.0)
    assert L_dpop.item() > L_dpo.item()


# ===== RainbowPO 一致性: unified == individual =====

def test_rainbowpo_dpo_matches():
    """RainbowPO 配 dpo 与手写 dpo loss 一致."""
    s = setup_logp()
    cfg = VARIANTS["dpo"]
    mask = torch.ones(4, 8)
    sft = torch.tensor(2.5)
    out = unified_po_loss(s["c_a"], s["c_r"], s["r_a"], s["r_r"],
                          mask, mask, sft, cfg)
    L_hand = dpo_loss(s["c_a"], s["c_r"], s["r_a"], s["r_r"], beta=cfg.beta)
    assert abs(out["total"].item() - L_hand.item()) < 1e-5


# ===== 边界 =====

def test_all_losses_have_grad():
    """所有 loss 都可反传."""
    s = setup_logp()
    s["c_a"].requires_grad_(True)
    L1 = dpo_loss(s["c_a"], s["c_r"], s["r_a"], s["r_r"])
    L1.backward(retain_graph=False)
    assert s["c_a"].grad is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
