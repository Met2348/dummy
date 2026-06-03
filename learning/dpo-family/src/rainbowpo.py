"""RainbowPO (Pal 2024) — 7 个 DPO 变体的统一公式.

unified loss = α · L_pref + (1-α) · L_NLL
其中 L_pref = -f(margin), f 由超参选择.

7 个变体 = 4 维超参组合:
    1. use_ref     : 是否用 ref model (DPO=T, ORPO/SimPO/CPO=F)
    2. length_norm : 是否 length normalize (SimPO=T)
    3. loss_type   : sigmoid | squared | hinge
    4. add_sft     : 是否加 NLL on chosen (ORPO/CPO=T)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class POConfig:
    name: str
    use_ref: bool
    length_norm: bool
    loss_type: str   # "sigmoid" | "squared" | "hinge"
    add_sft: bool
    beta: float = 0.1
    lambda_sft: float = 1.0


VARIANTS = {
    "dpo":   POConfig("dpo",   True,  False, "sigmoid", False),
    "ipo":   POConfig("ipo",   True,  False, "squared", False),
    "orpo":  POConfig("orpo",  False, False, "sigmoid", True, beta=1.0, lambda_sft=10.0),  # SFT 主导
    "simpo": POConfig("simpo", False, True,  "sigmoid", False, beta=2.5),
    "cpo":   POConfig("cpo",   False, False, "sigmoid", True, lambda_sft=2.0),
    "kto":   POConfig("kto",   True,  False, "sigmoid", False),   # 单边，特殊处理
    "dpop":  POConfig("dpop",  True,  False, "sigmoid", False),
}


def _pref_loss(margin: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "sigmoid":
        return -F.logsigmoid(margin).mean()
    if loss_type == "squared":
        # IPO style: minimize (margin - target)^2
        target = 0.5
        return ((margin - target) ** 2).mean()
    if loss_type == "hinge":
        return F.relu(1.0 - margin).mean()
    raise ValueError(loss_type)


def unified_po_loss(
    log_p_c_actor: torch.Tensor, log_p_c_ref: torch.Tensor,
    log_p_r_actor: torch.Tensor, log_p_r_ref: torch.Tensor,
    mask_c: torch.Tensor, mask_r: torch.Tensor,
    sft_loss_chosen: torch.Tensor,
    cfg: POConfig,
) -> dict:
    """统一接口 — 一个函数算所有 7 个变体."""
    if cfg.use_ref:
        log_ratio_c = log_p_c_actor - log_p_c_ref
        log_ratio_r = log_p_r_actor - log_p_r_ref
    else:
        log_ratio_c = log_p_c_actor
        log_ratio_r = log_p_r_actor

    if cfg.length_norm:
        lc = mask_c.sum(dim=1).clamp(min=1)
        lr = mask_r.sum(dim=1).clamp(min=1)
        log_ratio_c = log_ratio_c / lc
        log_ratio_r = log_ratio_r / lr

    margin = cfg.beta * (log_ratio_c - log_ratio_r)
    L_pref = _pref_loss(margin, cfg.loss_type)
    L_total = L_pref + cfg.lambda_sft * sft_loss_chosen if cfg.add_sft else L_pref
    return {"total": L_total, "pref": L_pref, "margin_mean": margin.mean()}


if __name__ == "__main__":
    print("RainbowPO — 7 变体统一公式 smoke test\n" + "=" * 50)
    torch.manual_seed(0)
    B, T = 4, 8
    log_p_c_a = (torch.randn(B, T) - 2).sum(dim=1)
    log_p_c_r = (torch.randn(B, T) - 2).sum(dim=1)
    log_p_r_a = (torch.randn(B, T) - 3).sum(dim=1)
    log_p_r_r = (torch.randn(B, T) - 3).sum(dim=1)
    mask_c = torch.ones(B, T)
    mask_r = torch.ones(B, T)
    sft_l = torch.tensor(2.5)
    print(f"{'name':6s} {'use_ref':8s} {'len_norm':9s} {'loss':9s} {'sft':5s} {'L_total':10s} {'margin':8s}")
    for name in ["dpo", "ipo", "orpo", "simpo", "cpo", "dpop"]:
        cfg = VARIANTS[name]
        out = unified_po_loss(log_p_c_a, log_p_c_r, log_p_r_a, log_p_r_r,
                              mask_c, mask_r, sft_l, cfg)
        print(f"{name:6s} {str(cfg.use_ref):8s} {str(cfg.length_norm):9s} "
              f"{cfg.loss_type:9s} {str(cfg.add_sft):5s} "
              f"{out['total'].item():10.4f} {out['margin_mean'].item():+8.4f}")
    print("\n→ 一个 unified_po_loss + 一个 POConfig = 6 个 PO 变体的全部.")
