"""Router 负载均衡测试."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe_layer_naive import MoELayer       # noqa
from gshard_router import GShardRouter     # noqa
from switch_router import SwitchRouter     # noqa


def test_moe_forward_shape():
    torch.manual_seed(0)
    layer = MoELayer(d_model=32, n_experts=4, top_k=2, d_ff=64)
    x = torch.randn(2, 16, 32)
    y, aux = layer(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.isfinite(aux)


def test_router_load_balance_after_training():
    """100 step 后，各 expert 利用率应在 1/k ± 30% 内."""
    torch.manual_seed(0)
    layer = MoELayer(d_model=16, n_experts=4, top_k=2, d_ff=32)
    opt = torch.optim.Adam(layer.parameters(), lr=1e-3)
    for _ in range(100):
        x = torch.randn(8, 16, 16)
        y, aux = layer(x)
        loss = y.pow(2).mean() + 0.01 * aux
        loss.backward()
        opt.step()
        opt.zero_grad()
    # 最终路由分布
    x_test = torch.randn(1, 32, 16)
    with torch.no_grad():
        logits = layer.gate(x_test.view(-1, 16))
        gates = F.softmax(logits, dim=-1)
        _, idx = gates.topk(2, dim=-1)
    util = torch.bincount(idx.flatten(), minlength=4).float()
    util = util / util.sum()
    ideal = 1 / 4
    # 每 expert 应在 0.15 ~ 0.35 (即 ±60%)
    for u in util.tolist():
        assert ideal * 0.4 <= u <= ideal * 1.6, f"util {u:.3f} 偏离 ideal {ideal}"


def test_gshard_aux_decreases_when_balanced():
    """随机均匀输入下 aux 应较小."""
    r = GShardRouter(d_model=16, n_experts=4)
    x = torch.randn(100, 16)
    _, _, aux, _ = r(x)
    assert aux.item() < 5.0, f"aux too large: {aux.item()}"


def test_switch_top1_one_expert_per_token():
    r = SwitchRouter(d_model=16, n_experts=4)
    x = torch.randn(20, 16)
    g, idx, _ = r(x)
    assert g.shape == (20, 1)
    assert idx.shape == (20, 1)
