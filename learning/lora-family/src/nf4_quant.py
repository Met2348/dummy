"""NF4 fake-quant 共享模块。

实现 QLoRA 论文 (Dettmers et al., 2023, arXiv:2305.14314) §3 的 NF4 算法。
用 PyTorch 纯实现 → 跨平台跑（CPU/GPU），不依赖 bitsandbytes。

Fake-quant 含义:
  - 前向：W → quantize → dequantize → W̃（值真的损失精度，与真量化等价）
  - 反向：STE（Straight-Through Estimator）梯度直接穿过

主要导出:
  - NF4_VALUES        16 个 NF4 网格点常数
  - nf4_quantize      W → (indices, absmax_scale)
  - nf4_dequantize    (indices, absmax_scale) → W̃
  - nf4_quant_dequant 组合 + STE
  - double_quantize_absmax / double_dequantize_absmax  二次量化（QLoRA §3.2）
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# NF4 lookup table: N(0, 1) 的 16 分位点
# 来源: QLoRA 论文公式 3 + bitsandbytes 参考实现
# 关键性质: 8 正、8 负、严格单调、对称（index 7 == 0.0）
NF4_VALUES = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=torch.float32,
)


def nf4_quantize(W: torch.Tensor, block_size: int = 64):
    """NF4 量化（blockwise）。

    步骤 (QLoRA 公式 3):
      1. 把 W flatten 并 reshape 成 (n_blocks, block_size)
      2. 每块: absmax = max(|W|)（per-block scale）
      3. normalize: W_norm = W / absmax ∈ [-1, 1]
      4. quantize: 把每个 W_norm 值映射到最近的 NF4 网格点 (in [0, 15])

    Returns:
        indices: torch.uint8, shape (n_blocks, block_size)
        absmax:  torch.float, shape (n_blocks,)
        orig_shape: 原始形状（用于 dequantize 恢复）
        pad:    pad 长度（用于 dequantize 裁剪）
    """
    orig_shape = W.shape
    flat = W.flatten().float()
    n_elems = flat.numel()
    pad = (block_size - n_elems % block_size) % block_size
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.view(-1, block_size)

    absmax = blocks.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    normalized = blocks / absmax  # ∈ [-1, 1]

    nf4 = NF4_VALUES.to(W.device)
    # 对每个值找最近的 NF4 网格点（broadcast 出 (n_blocks, block_size, 16) 距离）
    dist = (normalized.unsqueeze(-1) - nf4).abs()
    indices = dist.argmin(dim=-1).to(torch.uint8)

    return indices, absmax.squeeze(-1), orig_shape, pad


def nf4_dequantize(
    indices: torch.Tensor,
    absmax: torch.Tensor,
    orig_shape: torch.Size,
    pad: int,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """从 (indices, absmax) 反量化恢复 W̃。"""
    device = device or indices.device
    nf4 = NF4_VALUES.to(device).to(dtype)
    values = nf4[indices.long()]  # (n_blocks, block_size)
    W_hat = values * absmax.unsqueeze(-1)
    flat = W_hat.flatten()
    if pad:
        flat = flat[:-pad]
    return flat.view(orig_shape).to(dtype)


def nf4_quant_dequant(W: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """组合 quantize + dequantize，含 STE。

    Forward:  返回量化后的 W̃（与 W 数值上不同，模拟真量化损失）
    Backward: W̃ - W 部分被 detach，梯度直接穿过原始 W
    """
    indices, absmax, orig_shape, pad = nf4_quantize(W, block_size)
    W_hat = nf4_dequantize(indices, absmax, orig_shape, pad, W.device, W.dtype)
    return W + (W_hat - W).detach()


def double_quantize_absmax(
    absmax: torch.Tensor,
    block_size_outer: int = 256,
):
    """Double Quantization: 把 NF4 的 per-block absmax (fp32) 用 INT8 再量化。

    QLoRA 论文 §3.2: 把 64 个 fp32 absmax 用 256-block INT8 量化，省显存。

    简化实现：用对称 INT8 而非论文的 dynamic exponent，足以教学。
    """
    flat = absmax.flatten().float()
    n_elems = flat.numel()
    pad = (block_size_outer - n_elems % block_size_outer) % block_size_outer
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.view(-1, block_size_outer)
    outer_absmax = blocks.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    normalized = blocks / outer_absmax  # ∈ [-1, 1]
    indices = (normalized * 127).round().clamp(-128, 127).to(torch.int8)
    return indices, outer_absmax.squeeze(-1), absmax.shape, pad


def double_dequantize_absmax(
    indices: torch.Tensor,
    outer_absmax: torch.Tensor,
    orig_shape: torch.Size,
    pad: int,
) -> torch.Tensor:
    """反量化 double-quant 的 absmax。"""
    values = indices.float() / 127.0 * outer_absmax.unsqueeze(-1)
    flat = values.flatten()
    if pad:
        flat = flat[:-pad]
    return flat.view(orig_shape)


def main():
    """快速 demo。"""
    torch.manual_seed(42)
    W = torch.randn(256, 256)
    W_hat = nf4_quant_dequant(W, block_size=64)
    rmse = (W - W_hat).pow(2).mean().sqrt().item()
    print(f"NF4 fake-quant demo")
    print(f"  W shape: {tuple(W.shape)}")
    print(f"  RMSE:    {rmse:.4f}")
    print(f"  W max:   {W.abs().max().item():.4f}")
    print(f"  W̃ max:   {W_hat.abs().max().item():.4f}")


if __name__ == "__main__":
    main()
