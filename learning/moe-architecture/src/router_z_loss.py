"""Router z-loss + crash demo — ST-MoE 2022."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_layer_naive import MoELayer


def router_z_loss(logits: torch.Tensor) -> torch.Tensor:
    """防 logits 整体爆炸."""
    return torch.logsumexp(logits, dim=-1).pow(2).mean()


def crash_demo(use_aux: bool = False, steps: int = 200) -> list[list[float]]:
    """注入 crash — 不加 aux loss 时 router 会偏向 1 expert."""
    torch.manual_seed(0)
    layer = MoELayer(d_model=16, n_experts=4, top_k=2, d_ff=32)
    opt = torch.optim.Adam(layer.parameters(), lr=3e-3)
    layer.train()
    util_history = []
    for step in range(steps):
        x = torch.randn(64, 16, 16)
        y, aux = layer(x)
        loss = y.pow(2).mean()
        if use_aux:
            loss = loss + 0.01 * aux
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % 10 == 0:
            with torch.no_grad():
                gates = F.softmax(layer.gate(x.view(-1, 16)), dim=-1)
                _, idx = gates.topk(2, dim=-1)
                util = torch.bincount(idx.flatten(), minlength=4).float()
                util = (util / util.sum()).tolist()
                util_history.append(util)
    return util_history


if __name__ == "__main__":
    # logits
    logits = torch.randn(16, 8) * 5
    print(f"z_loss = {router_z_loss(logits).item():.4f}")
    print("\n=== Crash demo: no aux loss ===")
    h_no_aux = crash_demo(use_aux=False, steps=200)
    print(f"  step 0:   {[f'{u:.2f}' for u in h_no_aux[0]]}")
    print(f"  step 100: {[f'{u:.2f}' for u in h_no_aux[10]]}")
    print(f"  step 200: {[f'{u:.2f}' for u in h_no_aux[-1]]}")
    print("\n=== With aux loss ===")
    h_aux = crash_demo(use_aux=True, steps=200)
    print(f"  step 0:   {[f'{u:.2f}' for u in h_aux[0]]}")
    print(f"  step 200: {[f'{u:.2f}' for u in h_aux[-1]]}")
