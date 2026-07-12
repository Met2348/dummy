"""kv_cache_step 参考实现——见 dictation/specs/kv_cache_spec.py 的接口约定。
风格对齐 interview-prep/src/mlcoding/kv_cache.py 里 CachedSelfAttention.step 的缓存拼接部分。
"""
from __future__ import annotations

import torch


def kv_cache_step(
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    cache_k: torch.Tensor | None,
    cache_v: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cache_k is None:
        return new_k, new_v
    updated_k = torch.cat([cache_k, new_k], dim=2)
    updated_v = torch.cat([cache_v, new_v], dim=2)
    return updated_k, updated_v


if __name__ == "__main__":
    from dictation.checks.kv_cache_check import check
    check(kv_cache_step)
    print("[PASS] kv_cache_solution 自检通过")
