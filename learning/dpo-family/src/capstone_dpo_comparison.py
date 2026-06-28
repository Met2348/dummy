"""Capstone: 6 个 PO 方法横向 benchmark.

同基座 (Qwen2.5-0.5B 模拟) + 同数据 (Anthropic-HH mock) + 同 step.
输出:
    - 6 方法 loss 曲线
    - reward margin 曲线 (chosen - rejected)
    - chosen prob 变化 (DPOP 反例 — DPO 可能下降)
    - length 漂移 (SimPO 优势 — 无漂移)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from rainbowpo import VARIANTS, unified_po_loss


def mock_step(cfg, init_state: dict, lr: float = 0.05) -> dict:
    """模拟一步：每个变体对 chosen/rejected log_p 更新方向不同."""
    s = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in init_state.items()}
    s["log_p_c_a"].requires_grad_(True)
    s["log_p_r_a"].requires_grad_(True)
    out = unified_po_loss(
        s["log_p_c_a"], s["log_p_c_r"], s["log_p_r_a"], s["log_p_r_r"],
        s["mask_c"], s["mask_r"], s["sft_l"], cfg,
    )
    out["total"].backward()
    with torch.no_grad():
        s["log_p_c_a"] -= lr * s["log_p_c_a"].grad
        s["log_p_r_a"] -= lr * s["log_p_r_a"].grad
    return {k: (v.detach() if hasattr(v, "detach") else v) for k, v in s.items()} | {
        "loss": out["total"].item(), "margin": out["margin_mean"].item()}


def benchmark(steps: int = 50):
    torch.manual_seed(0)
    B, T = 16, 12
    init = dict(
        log_p_c_a=(torch.randn(B) - 5),
        log_p_c_r=(torch.randn(B) - 5),
        log_p_r_a=(torch.randn(B) - 5),
        log_p_r_r=(torch.randn(B) - 5),
        mask_c=torch.ones(B, T),
        mask_r=torch.ones(B, T),
        sft_l=torch.tensor(2.5),
    )

    history = {n: {"loss": [], "margin": [], "chosen_prob": []} for n in
               ["dpo", "ipo", "orpo", "simpo", "cpo", "dpop"]}

    for name in history:
        state = init
        cfg = VARIANTS[name]
        for _ in range(steps):
            state = mock_step(cfg, state)
            history[name]["loss"].append(state["loss"])
            history[name]["margin"].append(state["margin"])
            history[name]["chosen_prob"].append(state["log_p_c_a"].mean().item())
    return history


def print_table(history: dict):
    print(f"\n{'method':8s} {'final_loss':12s} {'final_margin':14s} {'Δ chosen_logp':16s}")
    print("-" * 60)
    for name, h in history.items():
        L = h["loss"][-1]
        M = h["margin"][-1]
        d = h["chosen_prob"][-1] - h["chosen_prob"][0]
        marker = " ⚠️ 下降" if d < 0 else ""
        print(f"{name:8s} {L:12.4f} {M:+14.4f} {d:+16.4f}{marker}")


if __name__ == "__main__":
    print("Capstone — 6 PO 变体横向 benchmark\n" + "=" * 60)
    h = benchmark(steps=50)
    print_table(h)
    print("\n观察:")
    print("  - DPO: margin 持续涨，但 chosen_logp 可能下降 (反例)")
    print("  - DPOP: hinge 强制 chosen_logp 不降")
    print("  - SimPO: 无 ref, 训练快，margin 稳定")
    print("  - ORPO: SFT 主导，最适合小数据")
