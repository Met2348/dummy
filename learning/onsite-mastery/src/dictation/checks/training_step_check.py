"""check(target) for training_step —— 纯断言，独立于 solutions/，不导入参考实现。

验证点：
1. 在一个简单可拟合的 toy 回归任务上连续跑很多步，loss 应大幅下降。
2. 返回值就是"这一步 backward 之前、当前参数下"的 loss，和独立算的参考值一致。
3. zero_grad 确实生效：故意在调用前塞一份"脏"梯度，调用后梯度应该只反映这一步的
   真实反传结果，而不是脏梯度 + 真实梯度的累加。
4. 梯度裁剪确实生效：人为构造巨大梯度，调用后 model 参数的整体梯度范数应 <= clip_norm。
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _total_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.pow(2).sum().item()
    return total ** 0.5


def check(target) -> None:
    torch.manual_seed(0)

    # ---- 1) & 2) toy 回归任务：loss 应大幅下降 + 返回值应等于当前参数下的 loss ----
    d_in, d_out = 6, 2
    model = nn.Linear(d_in, d_out)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    x = torch.randn(32, d_in)
    true_w = torch.randn(d_in, d_out)
    y = x @ true_w

    losses = []
    for step in range(200):
        with torch.no_grad():
            ref_loss = loss_fn(model(x), y).item()
        returned = target(model, optimizer, (x, y), loss_fn, clip_norm=10.0)
        if step == 0:
            assert abs(returned - ref_loss) < 1e-4, (
                f"返回值应是 backward 前当前参数下的 loss: 参考={ref_loss}, 实际返回={returned}"
            )
        losses.append(returned)

    assert losses[-1] < 0.3 * losses[0], f"loss 应大幅下降: {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert all(torch.isfinite(torch.tensor(losses)))

    # ---- 3) zero_grad 确实生效 ----
    model2 = nn.Linear(d_in, d_out)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    loss_fn2 = nn.MSELoss()
    batch2 = (torch.randn(16, d_in), torch.randn(16, d_out))

    # 第一步先跑一次，拿到"干净"的参考梯度（同样的模型状态、同样的 batch）
    model_ref = nn.Linear(d_in, d_out)
    model_ref.load_state_dict(model2.state_dict())
    optimizer_ref = torch.optim.SGD(model_ref.parameters(), lr=0.01)
    optimizer_ref.zero_grad()
    loss_fn2(model_ref(batch2[0]), batch2[1]).backward()
    ref_grad_norm = _total_grad_norm(model_ref)

    # 故意在真正调用前塞一份很大的"脏"梯度，模拟"忘记 zero_grad 的历史遗留"
    for p in model2.parameters():
        p.grad = torch.full_like(p, 1000.0)

    target(model2, optimizer2, batch2, loss_fn2, clip_norm=1e6)  # clip_norm 设很大，不干扰这个检验
    polluted_grad_norm_estimate = ref_grad_norm + 1000.0 * sum(
        p.numel() for p in model2.parameters()
    ) ** 0.5
    actual_norm = _total_grad_norm(model2)
    assert actual_norm < 0.5 * polluted_grad_norm_estimate, (
        f"梯度范数({actual_norm:.3f})看起来像是脏梯度(~1000量级)和真实梯度叠加了，"
        f"疑似没有调用 zero_grad()"
    )
    assert abs(actual_norm - ref_grad_norm) < 1e-3, (
        f"清零后应得到和独立参考一致的梯度范数: 参考={ref_grad_norm}, 实际={actual_norm}"
    )

    # ---- 4) 梯度裁剪确实生效 ----
    model3 = nn.Linear(d_in, d_out)
    optimizer3 = torch.optim.SGD(model3.parameters(), lr=0.01)
    # loss_fn 里乘一个很大的系数，人为放大梯度
    huge_loss_fn = lambda pred, y: 1e6 * ((pred - y) ** 2).mean()  # noqa: E731
    batch3 = (torch.randn(16, d_in), torch.randn(16, d_out))
    clip_norm = 1.0
    target(model3, optimizer3, batch3, huge_loss_fn, clip_norm=clip_norm)
    clipped_norm = _total_grad_norm(model3)
    assert clipped_norm <= clip_norm + 1e-3, (
        f"梯度裁剪未生效: clip_norm={clip_norm}, 实际梯度范数={clipped_norm}"
    )
    assert clipped_norm > 1e-8, "裁剪之后梯度不应该是 0（不是没算梯度，是裁剪到了 clip_norm 附近）"
