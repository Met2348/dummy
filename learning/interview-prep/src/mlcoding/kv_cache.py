"""KV Cache 增量解码，从零手写并对拍全量前向。

面试高频度 ★★★★。得分点：因果注意力下 token t 只依赖 <=t 的 K/V，历史 K/V 不变
→ 可缓存复用，把每步 O(T^2) 降到 O(T)。核心不变量：逐 token 走缓存 == 一次喂全序列。
只有 decoder（因果）能用；encoder（双向）不能。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class CachedSelfAttention(nn.Module):
    """单头因果自注意力，支持全量 forward 与带缓存的单步 step。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.d = d_model
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """全量：x (B,T,D) -> (B,T,D)，因果掩码。"""
        b, t, _ = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)
        mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))
        return scores.softmax(-1) @ v

    def step(self, x_t: torch.Tensor, cache: dict | None):
        """单步：x_t (B,1,D)。cache 存历史 K,V。返回 (out(B,1,D), new_cache)。"""
        q, k, v = self.q(x_t), self.k(x_t), self.v(x_t)      # 各 (B,1,D)
        if cache is not None:
            k = torch.cat([cache["k"], k], dim=1)            # 拼历史 K
            v = torch.cat([cache["v"], v], dim=1)
        new_cache = {"k": k, "v": v}
        # 当前 query 对所有历史 key（含自己）做注意力；无需掩码（未来还不存在）
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)  # (B,1,Tk)
        out = scores.softmax(-1) @ v                          # (B,1,D)
        return out, new_cache


def _self_test() -> None:
    torch.manual_seed(0)
    attn = CachedSelfAttention(d_model=16).eval()
    x = torch.randn(2, 6, 16)

    with torch.no_grad():
        full = attn(x)                                        # 全量
        cache, outs = None, []
        for t in range(x.size(1)):                            # 逐 token 走缓存
            o, cache = attn.step(x[:, t:t + 1], cache)
            outs.append(o)
        stepwise = torch.cat(outs, dim=1)

    # 核心不变量：缓存版逐元素等于全量版
    err = (full - stepwise).abs().max().item()
    assert err < 1e-5, f"缓存与全量不一致 err={err}"

    # 缓存确实在增长（第 t 步 K 长度 = t+1）
    assert cache["k"].shape[1] == x.size(1)
    print(f"[PASS] kv_cache: 逐token缓存==全量前向 (err={err:.2e})")


if __name__ == "__main__":
    _self_test()
