"""bitsandbytes-style nf4 quantization (mock, quantile-based)."""
from __future__ import annotations

import torch


# NF4 quantile-derived codes from the bitsandbytes paper (16 levels)
NF4_CODES = torch.tensor([
    -1.0,    -0.6962,  -0.5251,  -0.3949,
    -0.2844, -0.1848,  -0.0911,   0.0,
     0.0796, 0.1609,    0.2461,    0.3379,
     0.4407, 0.5626,    0.7230,    1.0,
])


def quantize_nf4(W: torch.Tensor, block_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-block nf4 quantization: each `block_size` elements share a scale."""
    flat = W.flatten()
    pad = (-len(flat)) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
    blocks = flat.reshape(-1, block_size)
    scales = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-9)
    normalised = blocks / scales
    # quantize each value to nearest NF4 code
    idx = (normalised.unsqueeze(-1) - NF4_CODES).abs().argmin(dim=-1)
    return idx.to(torch.uint8), scales.squeeze(-1)


def dequantize_nf4(idx: torch.Tensor, scales: torch.Tensor, block_size: int = 64, orig_shape: torch.Size = None) -> torch.Tensor:
    codes = NF4_CODES[idx.long()]   # [blocks, block_size]
    blocks = codes * scales.unsqueeze(-1)
    flat = blocks.reshape(-1)
    if orig_shape is not None:
        n = int(torch.tensor(orig_shape).prod().item())
        flat = flat[:n]
        return flat.reshape(orig_shape)
    return flat
