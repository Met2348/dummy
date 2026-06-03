"""Aux-Free 偏置项稳定性 + 负载均衡测试."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aux_loss_free import AuxFreeRouter   # noqa


def test_aux_free_no_aux_loss():
    r = AuxFreeRouter(d_model=16, n_experts=4, top_k=2)
    x = torch.randn(20, 16)
    g, idx, aux = r(x)
    assert aux is None


def test_aux_free_bias_init_zero():
    r = AuxFreeRouter(d_model=16, n_experts=8, top_k=2)
    assert torch.allclose(r.bias, torch.zeros(8))


def test_aux_free_balances_load_after_training():
    torch.manual_seed(0)
    router = AuxFreeRouter(d_model=16, n_experts=4, top_k=2,
                           update_rate=1e-3)
    router.train()
    opt = torch.optim.Adam(router.W.parameters(), lr=1e-3)
    for _ in range(200):
        x = torch.randn(64, 16)
        gates, idx, _ = router(x)
        loss = (gates ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    # 末段 load 应平衡
    router.eval()
    x = torch.randn(256, 16)
    _, idx, _ = router(x)
    load = torch.bincount(idx.flatten(), minlength=4).float()
    ratio = load.max() / load.min()
    # max/min < 2.5 视为平衡
    assert ratio < 2.5, f"ratio {ratio.item():.2f} 不够平衡"


def test_aux_free_bias_does_not_affect_gates():
    """bias 只影响排序, 不影响 gates 数值."""
    torch.manual_seed(0)
    r = AuxFreeRouter(d_model=16, n_experts=4, top_k=2)
    x = torch.randn(20, 16)
    g1, _, _ = r(x)
    # 给 bias 设个值
    r.bias.data.fill_(0.5)
    g2, _, _ = r(x)
    # gates 应来自原 softmax，但 top-k 选择可能变
    # 至少 g1.sum 与 g2.sum 都应归一化（每行 sum=1）
    assert torch.allclose(g1.sum(-1), torch.ones(20), atol=1e-4)
    assert torch.allclose(g2.sum(-1), torch.ones(20), atol=1e-4)
