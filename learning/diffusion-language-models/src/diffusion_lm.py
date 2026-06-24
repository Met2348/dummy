"""
diffusion_lm.py — masked diffusion 语言模型 (dLLM) 最小实现 + AR 对照.

为什么需要它 (M13.6): 生成式媒体绕一圈回 NLP 本行 —— **文本也能扩散**。主流 LLM 是自回归
(AR, 一次一个 token, 左到右)。dLLM (如 LLaDA) 改用**离散/masked 扩散**: 前向逐步把 token 换成
[MASK], 模型学「给定部分遮盖序列, 预测被遮位」; 生成 = 从全 [MASK] 出发, **按置信度并行迭代
解码** (非自回归, T≪L 轮)。这是 2025-2026 的范式迁移热点。

本文件玩具「语言」: 长度 L 的**回文** [a,b,c,c,b,a] (位置 i 必须等于位置 L-1-i)。为什么选回文?
它让 dLLM 的核心优势**可证伪**: 挖掉中间靠左的位 (镜像在右侧), AR 因果注意力**看不到右侧**→
只能瞎猜; dLLM 看双侧→准。这是 dLLM「双向/可控生成」卖点的最小可验证版。

纯 torch (tiny CPU), 确定性。同文件含一个 AR 对照模型 (N2 用)。
"""
from __future__ import annotations

import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

V = 6           # 词表 (token 0..5)
L = 6           # 序列长度 (回文)
MASK = V        # [MASK] 的 id (= 6)
BOS = V         # AR 的起始符 (AR 词表里复用 id 6)


# ───────────────────────── 数据: 回文序列 ─────────────────────────
def make_sequences(n: int = 2000, seed: int = 0) -> np.ndarray:
    """n 条回文序列 (n, L): 前半随机, 后半镜像。合法 = 回文。"""
    rng = np.random.default_rng(seed)
    half = (L + 1) // 2
    left = rng.integers(0, V, size=(n, half))
    seqs = np.concatenate([left, left[:, ::-1][:, L % 2:]], axis=1)
    return seqs.astype(np.int64)        # (n, L)


def is_palindrome(seqs: np.ndarray) -> float:
    """合法率 = 是回文的比例。"""
    seqs = np.asarray(seqs)
    return float(np.mean([np.array_equal(s, s[::-1]) for s in seqs]))


# ───────────────────────── dLLM: masked diffusion ─────────────────────────
def build_dlm(d_model: int = 64, seed: int = 0):
    """masked diffusion LM: 双向 transformer, 输入含 [MASK], 每位预测 V 个 token 的 logits。"""
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        print(f"[diffusion_lm] 无 torch ({exc!r})"); return None
    torch.manual_seed(seed)

    class DLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V + 1, d_model)          # +1 给 [MASK]
            self.pos = nn.Parameter(torch.randn(1, L, d_model) * 0.02)
            self.enc = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, 4, d_model * 2, batch_first=True), 2)
            self.out = nn.Linear(d_model, V)

        def forward(self, x):                                 # x:(B,L) 含 MASK, 无因果掩码=双向
            h = self.emb(x) + self.pos
            return self.out(self.enc(h))                      # (B,L,V)

    return DLM()


def train_dlm(model, data: np.ndarray, epochs: int = 600, lr: float = 3e-3, seed: int = 0):
    """训练: 随机比例遮盖, 只在被遮位算 CE (预测原 token)。"""
    import torch
    import torch.nn as nn
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    x = torch.tensor(data)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        ratio = rng.uniform(0.2, 1.0)                         # 每步随机遮盖比例 (扩散的"噪声水平")
        mask = torch.tensor(rng.random(data.shape) < ratio)
        xin = x.clone(); xin[mask] = MASK
        logits = model(xin)
        loss = nn.functional.cross_entropy(logits[mask], x[mask])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses


def generate_dlm(model, n: int = 100, rounds: int = 3, seed: int = 0, record: bool = False):
    """从全 [MASK] 并行迭代解码: 每轮预测所有遮位, 提交最自信的若干位, 重复。非自回归。"""
    import torch
    rng = np.random.default_rng(seed)
    x = torch.full((n, L), MASK, dtype=torch.long)
    traj = [x[0].clone().numpy()] if record else None
    per_round = int(np.ceil(L / rounds))
    with torch.no_grad():
        for r in range(rounds):
            logits = model(x)
            probs = torch.softmax(logits, -1)
            conf, pred = probs.max(-1)                        # 每位最自信 token + 置信度
            conf = conf.masked_fill(x != MASK, -1)            # 已定位不再考虑
            k = min(per_round if r < rounds - 1 else L, L)    # 本轮要提交的总位数上限
            for b in range(n):
                masked_idx = (x[b] == MASK).nonzero().flatten()
                if len(masked_idx) == 0:
                    continue
                order = masked_idx[torch.argsort(conf[b, masked_idx], descending=True)]
                take = order[:per_round] if r < rounds - 1 else order
                x[b, take] = pred[b, take]
            if record:
                traj.append(x[0].clone().numpy())
    return (x.numpy(), traj) if record else x.numpy()


def infill_dlm(model, partial: np.ndarray) -> np.ndarray:
    """双向 infilling: partial 含 MASK 的位由模型一次性填 (看左右全文)。"""
    import torch
    x = torch.tensor(partial).clone()
    if x.dim() == 1:
        x = x[None]
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(-1)
    out = x.clone()
    m = (x == MASK)
    out[m] = pred[m]
    return out.numpy()


# ───────────────────────── AR 对照模型 ─────────────────────────
def build_ar(d_model: int = 64, seed: int = 0):
    """自回归对照: 因果 transformer, 预测下一 token (左到右)。"""
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        print(f"[diffusion_lm] 无 torch ({exc!r})"); return None
    torch.manual_seed(seed)

    class AR(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V + 1, d_model)           # +1 给 BOS
            self.pos = nn.Parameter(torch.randn(1, L + 1, d_model) * 0.02)
            self.enc = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, 4, d_model * 2, batch_first=True), 2)
            self.out = nn.Linear(d_model, V)

        def forward(self, x):                                 # x:(B,T) 含 BOS 前缀
            import torch
            T = x.shape[1]
            h = self.emb(x) + self.pos[:, :T]
            cmask = torch.triu(torch.ones(T, T) * float('-inf'), 1)  # 因果掩码=单向
            return self.out(self.enc(h, mask=cmask))          # (B,T,V)

    return AR()


def train_ar(model, data: np.ndarray, epochs: int = 600, lr: float = 3e-3, seed: int = 0):
    """训练 AR: 输入 [BOS, t0..t_{L-1}], 预测 [t0..t_{L-1}]。"""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    x = torch.tensor(data)
    bos = torch.full((len(data), 1), BOS, dtype=torch.long)
    inp = torch.cat([bos, x], 1)                              # (B, L+1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        logits = model(inp[:, :-1])                           # 预测每个下一位
        loss = nn.functional.cross_entropy(logits.reshape(-1, V), x.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses


def generate_ar(model, n: int = 100, seed: int = 0) -> np.ndarray:
    """AR 左到右逐位生成 (L 步)。"""
    import torch
    torch.manual_seed(seed)
    x = torch.full((n, 1), BOS, dtype=torch.long)
    with torch.no_grad():
        for _ in range(L):
            logits = model(x)[:, -1]                          # 最后位的下一 token
            nxt = logits.argmax(-1, keepdim=True)
            x = torch.cat([x, nxt], 1)
    return x[:, 1:].numpy()                                   # 去掉 BOS


def ar_infill_accuracy(model, seqs: np.ndarray, blank_pos: int) -> float:
    """AR 对「中间靠左位」的 infilling: 只能用左侧上下文 (因果), 看准确率。"""
    import torch
    x = torch.tensor(seqs)
    bos = torch.full((len(seqs), 1), BOS, dtype=torch.long)
    inp = torch.cat([bos, x], 1)
    with torch.no_grad():
        # 预测 blank_pos 的分布 = 给定 [BOS, t0..t_{blank-1}] (右侧不可见)
        logits = model(inp[:, :blank_pos + 1])[:, -1]
        pred = logits.argmax(-1).numpy()
    return float((pred == seqs[:, blank_pos]).mean())


def dlm_infill_accuracy(model, seqs: np.ndarray, blank_pos: int) -> float:
    """dLLM 对同一位的 infilling: 看左右全文 (双向)。"""
    partial = seqs.copy(); partial[:, blank_pos] = MASK
    filled = infill_dlm(model, partial)
    return float((filled[:, blank_pos] == seqs[:, blank_pos]).mean())


if __name__ == "__main__":
    import torch
    data = make_sequences(2000, seed=0)
    print(f"回文数据 {data.shape}, 样例 {data[0]} (合法率 {is_palindrome(data):.2f})")

    dlm = build_dlm(); train_dlm(dlm, data, epochs=600)
    gen, traj = generate_dlm(dlm, n=200, rounds=3, seed=1, record=True)
    print(f"\ndLLM 并行生成 (3 轮): 合法回文率 {is_palindrome(gen):.2f}")
    print("  一条序列的解码轨迹 (6=MASK):")
    for i, s in enumerate(traj):
        print(f"    轮{i}: {s}")

    ar = build_ar(); train_ar(ar, data, epochs=600)
    gen_ar = generate_ar(ar, n=200, seed=1)
    print(f"\nAR 逐位生成 (6 步): 合法回文率 {is_palindrome(gen_ar):.2f}")

    # 双向优势: infill 位置 1 (镜像在右侧位置 4)
    test = make_sequences(300, seed=9)
    bp = 1
    print(f"\nInfilling 位置 {bp} (镜像在右侧, 由右文决定):")
    print(f"  dLLM (双向): 准确率 {dlm_infill_accuracy(dlm, test, bp):.2f}")
    print(f"  AR   (单向): 准确率 {ar_infill_accuracy(ar, test, bp):.2f}  (≈瞎猜 1/V={1/V:.2f})")
    print("→ dLLM 并行解码 + 双向 infilling; AR 单向看不到右文。这就是 dLLM 对 NLP 的价值。")
