"""LoRA 线性层，从零手写（冻结基座 + 低秩旁路）。

面试高频度 ★★★★（你 PhD 也天天用）。得分点：ΔW = B·A，A(in×r) 随机、B(r×out) 初始化为 0
→ 起始等价原层；只训 A,B；缩放 alpha/r。基座权重 requires_grad=False。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 4, alpha: int = 8):
        super().__init__()
        assert r > 0
        self.base = base
        for p in self.base.parameters():          # 冻结基座
            p.requires_grad_(False)
        in_f, out_f = base.in_features, base.out_features
        self.A = nn.Parameter(torch.randn(in_f, r) * (1.0 / r ** 0.5))
        self.B = nn.Parameter(torch.zeros(r, out_f))    # 关键：B=0 → 初始等价原层
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A @ self.B) * self.scaling

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _self_test() -> None:
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    lora = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(3, 16)

    # 1) 初始等价原层（B=0）
    assert torch.allclose(lora(x), base(x), atol=1e-6)

    # 2) 仅 A,B 可训（16*4 + 4*8 = 96），基座冻结
    assert lora.trainable_params() == 16 * 4 + 4 * 8
    assert base.weight.requires_grad is False

    # 3) 一步梯度后输出改变；基座权重无梯度
    opt = torch.optim.SGD([p for p in lora.parameters() if p.requires_grad], lr=0.1)
    before = lora(x).detach().clone()
    loss = (lora(x) - 1.0).pow(2).mean()
    loss.backward()
    assert base.weight.grad is None, "基座不该有梯度"
    assert lora.B.grad is not None and lora.A.grad is not None
    opt.step()
    assert not torch.allclose(lora(x), before, atol=1e-6), "训练后应改变"
    print("[PASS] lora: 初始等价 + 仅低秩可训 + 基座冻结 + 训练生效")


if __name__ == "__main__":
    _self_test()
