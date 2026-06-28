"""
tiny_transformer.py — 可 hook 中间激活的最小 transformer (M12 受控基座).

为什么需要它 (M12.1): 机制可解释性要「逆向工程」网络内部。要教 probing/patching/circuits, 需要一个
**已知 ground truth + 能读取/干预中间激活**的受控模型 (真 gpt2 太大、电路未知, 适合做"真实"演示,
但教方法需要干净的玩具)。本文件是一个从零搭的小 transformer, 暴露每层的 residual stream / attention /
MLP 激活 (run_with_cache), 训练在一个**结构已知**的玩具任务上 (increment-mod-V: 序列下一个=当前+1 mod V)。

这样后续专题能:
  - 12.2 probing: 从 residual 读出"当前值"概念
  - 12.3 patching: 把某位置激活替换, 看因果
  - 12.5 circuits: 看 attention 怎么搬运信息
纯 torch tiny CPU 确定性。真实模型演示用 learning/_shared/realmodels.py (gpt2)。
"""
from __future__ import annotations

import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

V = 12          # 词表 (token 0..11)
SEQ = 8         # 序列长度
D_MODEL = 32
N_LAYERS = 2
N_HEADS = 4


def make_data(n: int = 2000, seed: int = 0):
    """玩具任务: increment-mod-V 序列。每条序列从随机起点开始, 每位 = 前一位 + 1 (mod V)。
    模型要学「下一个 token = 当前 + 1 mod V」(结构已知, 便于解剖)。返回 (inputs, targets)。"""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, V, size=n)
    seqs = (starts[:, None] + np.arange(SEQ + 1)[None, :]) % V    # (n, SEQ+1)
    return seqs[:, :SEQ].astype(np.int64), seqs[:, 1:SEQ + 1].astype(np.int64)


def build_model(seed: int = 0):
    """最小 transformer, 暴露中间激活 (run_with_cache)。"""
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        print(f"[tiny_transformer] 无 torch ({exc!r})"); return None
    torch.manual_seed(seed)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)
            self.mlp = nn.Sequential(nn.Linear(D_MODEL, D_MODEL * 4), nn.GELU(),
                                     nn.Linear(D_MODEL * 4, D_MODEL))
            self.ln1 = nn.LayerNorm(D_MODEL); self.ln2 = nn.LayerNorm(D_MODEL)

        def forward(self, x, cache, li):
            import torch
            T = x.shape[1]
            mask = torch.triu(torch.ones(T, T) * float('-inf'), 1)
            a, attn_w = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                                  attn_mask=mask, need_weights=True, average_attn_weights=False)
            x = x + a                                          # residual: 写入 attn 输出
            if cache is not None:
                cache[f"attn_out_{li}"] = a.detach()
                cache[f"attn_pattern_{li}"] = attn_w.detach()  # (B, heads, T, T)
            m = self.mlp(self.ln2(x))
            x = x + m                                          # residual: 写入 MLP 输出
            if cache is not None:
                cache[f"mlp_out_{li}"] = m.detach()
                cache[f"resid_post_{li}"] = x.detach()
            return x

    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(V, D_MODEL)
            self.pos = nn.Parameter(torch.randn(1, SEQ, D_MODEL) * 0.02)
            self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
            self.ln_f = nn.LayerNorm(D_MODEL)
            self.unembed = nn.Linear(D_MODEL, V)

        def forward(self, tokens, cache=None):
            import torch
            x = self.embed(tokens) + self.pos[:, :tokens.shape[1]]
            if cache is not None:
                cache["resid_pre"] = x.detach()
            for li, blk in enumerate(self.blocks):
                x = blk(x, cache, li)
            return self.unembed(self.ln_f(x))

        def run_with_cache(self, tokens):
            cache = {}
            logits = self(tokens, cache=cache)
            return logits, cache

    return TinyTransformer()


def train(model, inputs, targets, epochs: int = 800, lr: float = 2e-3, seed: int = 0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    X = torch.tensor(inputs); Y = torch.tensor(targets)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        logits = model(X)
        loss = nn.functional.cross_entropy(logits.reshape(-1, V), Y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses


def accuracy(model, inputs, targets):
    import torch
    with torch.no_grad():
        pred = model(torch.tensor(inputs)).argmax(-1).numpy()
    return float((pred == targets).mean())


if __name__ == "__main__":
    import torch
    Xi, Yi = make_data(2000, seed=0)
    model = build_model()
    if model is not None:
        losses = train(model, Xi, Yi, epochs=800)
        acc = accuracy(model, *make_data(500, seed=9))
        print(f"tiny transformer 训练: loss {losses[0]:.3f} → {losses[-1]:.3f}, 测试准确率 {acc:.2f}")
        logits, cache = model.run_with_cache(torch.tensor(Xi[:1]))
        print("可读取的激活:", list(cache.keys()))
        print(f"residual stream 形状: {cache['resid_post_1'].shape} (B, T, d_model)")
        print(f"attention pattern 形状: {cache['attn_pattern_0'].shape} (B, heads, T, T)")
        print("→ 受控玩具 transformer 学会 increment-mod-V, 中间激活全可读取 (M12 解剖基座)。")
