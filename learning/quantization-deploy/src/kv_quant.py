"""KV cache int8 quantization — per-token symmetric scale."""
from __future__ import annotations

import torch


def quantize_kv_per_token(kv: torch.Tensor, n_bits: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize KV `[seq, n_heads, head_dim]` with one scale per token."""
    qmax = (1 << (n_bits - 1)) - 1
    seq = kv.shape[0]
    # collapse head + dim dimensions for scale computation
    flat = kv.reshape(seq, -1)
    scale = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-9) / qmax
    q = (flat / scale).round().clamp(-qmax, qmax).to(torch.int8)
    return q.reshape(kv.shape), scale.reshape(seq, 1, 1)


def dequantize_kv_per_token(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale


def attention_with_quant_kv(
    Q: torch.Tensor,            # [seq_q, n_heads, d_h]
    Kq: torch.Tensor,           # quantized K
    Vq: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> torch.Tensor:
    K = dequantize_kv_per_token(Kq, k_scale)
    V = dequantize_kv_per_token(Vq, v_scale)
    scale = 1.0 / (Q.shape[-1] ** 0.5)
    scores = torch.einsum("qhd,khd->qkh", Q, K) * scale
    attn = torch.softmax(scores, dim=1)
    return torch.einsum("qkh,khd->qhd", attn, V)
