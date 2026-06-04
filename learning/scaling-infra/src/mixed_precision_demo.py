"""Mixed precision demo - bf16 vs fp16 范围 + loss spike 检测."""
from __future__ import annotations

import torch


def precision_ranges() -> None:
    info = [
        ("fp32", torch.finfo(torch.float32)),
        ("bf16", torch.finfo(torch.bfloat16)),
        ("fp16", torch.finfo(torch.float16)),
    ]
    print("=== 浮点范围 ===")
    for name, f in info:
        print(f"  {name}: min={f.min:.3e} max={f.max:.3e} "
              f"eps={f.eps:.3e}")


def amp_training_template():
    print("\n=== AMP 训练模板 ===")
    print("""
import torch
from torch.amp import autocast

model = MyModel().cuda()
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-8)

for batch in loader:
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(batch.x)
        loss = criterion(out, batch.y)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
""")


def loss_spike_guard(loss: float, ema: float, threshold: float = 5.0) -> bool:
    """True if 应跳过这一步."""
    return loss > threshold * ema


def grad_norm_warn(grad_norm: float) -> str:
    if grad_norm > 100:
        return "DANGER: grad explosion"
    if grad_norm > 10:
        return "warn: grad high"
    if grad_norm < 1e-4:
        return "warn: grad vanishing"
    return "ok"


if __name__ == "__main__":
    precision_ranges()
    amp_training_template()

    print("\n=== loss spike guard ===")
    ema = 2.0
    for L in [2.1, 3.5, 15.0, 8.0]:
        skip = loss_spike_guard(L, ema)
        print(f"  loss={L} ema={ema}  → skip={skip}")

    print("\n=== grad norm warn ===")
    for g in [0.05, 1.5, 12, 250]:
        print(f"  grad_norm={g}: {grad_norm_warn(g)}")
