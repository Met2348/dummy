"""
circuits.py — 电路分析 + induction head 检测 + 逐头归因 (M12.5). 重度用真实 gpt2.

为什么需要它 (M12.5): 把单义特征 (M12.4) 连成**算法 (circuit)**。最经典的 circuit 是 **induction head**:
它实现「看到 AB...A 就预测 B」(in-context 复制), 是 in-context learning 的机制基础。真实 gpt2 预训练
出了 induction head —— 我们直接在 gpt2 上**找到它 + 因果验证**, 这是最有说服力的 mech interp 演示。

induction 检测: 喂一个「随机 token 序列重复两遍」, induction head 在第二遍某 token 处, 会 attend 到
第一遍该 token 的**下一个** token (因为「下一个就该是它」)。逐 (层,头) 算这个 attention = induction 分数。
因果验证: 消融最强 induction head, 模型预测重复的能力 (induction loss) 应变差。

纯 transformers gpt2 CPU 离线。需 output_attentions (eager)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_SHARED = Path(__file__).resolve().parents[2] / "_shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))


def make_repeated_tokens(tok, n_unique: int = 24, seed: int = 0):
    """随机 token 序列重复两遍 [BOS, r1..rk, r1..rk]。返回 token id 张量 (1, 2k+1)。"""
    import torch
    rng = np.random.default_rng(seed)
    # 从常见 token 里采 (避开特殊符), gpt2 词表 50257
    rand = rng.integers(100, 40000, size=n_unique)
    ids = np.concatenate([[tok.bos_token_id if tok.bos_token_id else 50256], rand, rand])
    return torch.tensor(ids[None], dtype=torch.long), n_unique


def induction_scores(model, tokens, n_unique: int):
    """逐 (层, 头) 的 induction 分数: 第二遍位置 attend 到「第一遍对应 token 的下一个」的平均注意力。
    返回 (n_layers, n_heads) 矩阵。"""
    import torch
    with torch.no_grad():
        out = model(tokens, output_attentions=True)
    atts = out.attentions                 # tuple(L), 每个 (1, heads, seq, seq)
    L = len(atts); H = atts[0].shape[1]
    seq = tokens.shape[1]
    half = n_unique
    scores = np.zeros((L, H))
    # 第二遍的位置: 1+half .. 1+2half-1; 对应"第一遍下一个"的位置 = (i - half) + 1
    for li in range(L):
        a = atts[li][0]                   # (heads, seq, seq)
        for hi in range(H):
            vals = []
            for i in range(1 + half, seq):
                tgt = (i - half) + 1       # induction 应 attend 的位置
                if 0 <= tgt < seq:
                    vals.append(float(a[hi, i, tgt]))
            scores[li, hi] = np.mean(vals) if vals else 0.0
    return scores


def induction_loss(model, tokens, n_unique: int, ablate=None):
    """模型预测「第二遍重复」的交叉熵 (越低=induction 越好)。ablate=(layer,head) 则消融该头。"""
    import torch
    handle = None
    if ablate is not None:
        L, Hh = ablate
        head_dim = model.config.n_embd // model.config.n_head

        def hook(m, inp, out):
            o = out[0].clone()
            o[:, :, Hh * head_dim:(Hh + 1) * head_dim] = 0.0   # 置零该头的输出片段
            return (o,) + tuple(out[1:])
        handle = model.transformer.h[L].attn.register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits = model(tokens).logits[0]      # (seq, vocab)
        half = n_unique
        # 第二遍位置 i 的目标 = tokens[i+1] (= 重复, 可由 induction 预测)
        idx = list(range(1 + half, tokens.shape[1] - 1))
        tgt = tokens[0, [i + 1 for i in idx]]
        pred = logits[idx]
        loss = torch.nn.functional.cross_entropy(pred, tgt).item()
    finally:
        if handle is not None:
            handle.remove()
    return loss


def per_head_ablation(model, tokens, n_unique: int):
    """逐头消融对 induction loss 的损害 (越大=该头对 induction 越关键)。返回 (L, H)。"""
    base = induction_loss(model, tokens, n_unique)
    L = model.config.n_layer; H = model.config.n_head
    grid = np.zeros((L, H))
    for li in range(L):
        for hi in range(H):
            grid[li, hi] = induction_loss(model, tokens, n_unique, ablate=(li, hi)) - base
    return grid, base


if __name__ == "__main__":
    import realmodels as rm
    tok, model = rm.gpt2(output_attentions=True)
    if model is not None:
        tokens, k = make_repeated_tokens(tok, n_unique=24, seed=0)
        scores = induction_scores(model, tokens, k)
        best = np.unravel_index(np.argmax(scores), scores.shape)
        print(f"induction 分数最高的头: 层{best[0]} 头{best[1]} (分数 {scores[best]:.2f})")
        print(f"  (这是 gpt2 的一个 induction head: 它 attend 到'上次出现的下一个 token')")
        base = induction_loss(model, tokens, k)
        abl = induction_loss(model, tokens, k, ablate=best)
        print(f"induction loss: 完整 {base:.2f} → 消融该头 {abl:.2f} "
              f"({'变差✓ 因果确认' if abl > base else '没变差'})")
        print("→ 找到 induction head + 消融它使预测重复变差 = 因果确认这个 circuit (M12.3 干预)。")
    else:
        print("无 gpt2, 跳过")
