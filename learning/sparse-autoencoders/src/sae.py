"""
sae.py — 稀疏自编码器 (SAE) / 字典学习, 把叠加特征解开成单义特征 (M12.4).

为什么需要它 (M12.4): superposition (M12.1-L2) 让神经元多义 (一个神经元混多个概念), 没法直接读。
**SAE (sparse autoencoder)** 是 Anthropic 的「显微镜」: 学一个**过完备 + 稀疏**的字典, 把激活分解成
许多稀疏激活的特征。因为特征数 >> 神经元数 + 稀疏约束, 每个学到的特征更可能**单义** (对应一个干净概念)。

机制 (逐项):
  encode: f = ReLU(W_enc · (x - b_dec) + b_enc)     稀疏特征码 (n_features 维, 远多于 d_in)
  decode: x̂ = W_dec · f + b_dec                     重建
  loss  = ||x - x̂||²  (重建)  +  λ·||f||₁  (稀疏: 逼大多数特征=0)
单义性来自: 过完备 (够多特征装下每个概念) + L1 稀疏 (每次只少数特征激活)。

通用 (输入任意激活矩阵), 既能用在玩具 transformer (有 ground truth), 也能用在真实 gpt2 激活上。
纯 torch tiny CPU 确定性。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_M121 = Path(__file__).resolve().parents[2] / "interp-foundations" / "src"
if str(_M121) not in sys.path:
    sys.path.insert(0, str(_M121))
import tiny_transformer as tt  # noqa: E402


def build_sae(d_in: int, n_features: int, seed: int = 0):
    """过完备稀疏自编码器 (n_features >> d_in)。"""
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        print(f"[sae] 无 torch ({exc!r})"); return None
    torch.manual_seed(seed)

    class SAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.b_dec = nn.Parameter(torch.zeros(d_in))
            self.W_enc = nn.Linear(d_in, n_features)
            self.W_dec = nn.Linear(n_features, d_in, bias=False)
            self.n_features = n_features

        def encode(self, x):
            import torch
            return torch.relu(self.W_enc(x - self.b_dec))

        def decode(self, f):
            return self.W_dec(f) + self.b_dec

        def forward(self, x):
            f = self.encode(x)
            return self.decode(f), f

    return SAE()


def train_sae(sae, acts: np.ndarray, epochs: int = 600, lr: float = 2e-3, l1: float = 2e-3, seed: int = 0):
    """训练 SAE: 重建损失 + L1 稀疏。返回 (losses, 稀疏度历史)。"""
    import torch
    torch.manual_seed(seed)
    X = torch.tensor(acts, dtype=torch.float32)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    losses, sparsity = [], []
    for _ in range(epochs):
        xhat, f = sae(X)
        recon = ((X - xhat) ** 2).mean()
        sparse = f.abs().mean()
        loss = recon + l1 * sparse
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(recon.item())
        sparsity.append(float((f > 1e-4).float().mean().item()))   # 平均激活特征比例
    return losses, sparsity


def feature_codes(sae, acts: np.ndarray):
    """返回每个样本的特征码 (n_samples, n_features)。"""
    import torch
    with torch.no_grad():
        f = sae.encode(torch.tensor(acts, dtype=torch.float32))
    return f.numpy()


def monosemanticity(codes: np.ndarray, labels: np.ndarray):
    """单义性: 对每个"活跃"特征, 看它主要在哪个 label 上激活, 及纯度 (该 label 占其激活的比例)。
    返回 (active_features, 平均纯度)。纯度高 = 特征单义 (一个特征一个概念)。"""
    purities, active = [], []
    for fi in range(codes.shape[1]):
        col = codes[:, fi]
        if (col > 1e-3).sum() < 3:
            continue                                 # 几乎不激活的特征跳过
        active.append(fi)
        # 该特征激活加权下, 各 label 的占比
        labs = np.unique(labels)
        mass = np.array([col[labels == l].sum() for l in labs])
        purities.append(float(mass.max() / (mass.sum() + 1e-8)))
    return active, (float(np.mean(purities)) if purities else 0.0)


def tiny_mlp_activations(model, inputs: np.ndarray, layer: int = 1):
    """取玩具 transformer 某层 MLP 输出 (最后位置) + 当前值 label。"""
    import torch
    _, cache = model.run_with_cache(torch.tensor(inputs))
    acts = cache[f"resid_post_{layer}"][:, -1, :].numpy()
    labels = inputs[:, -1]
    return acts, labels


if __name__ == "__main__":
    import torch
    # 玩具: SAE 解叠加, 特征应对应"当前值"
    Xi, Yi = tt.make_data(2000, seed=0)
    model = tt.build_model(); tt.train(model, Xi, Yi, epochs=800)
    acts, labels = tiny_mlp_activations(model, tt.make_data(1500, seed=2)[0])
    print(f"玩具激活 {acts.shape} (d_model={acts.shape[1]}), 概念类别 {tt.V} 个")

    sae = build_sae(d_in=acts.shape[1], n_features=tt.V * 3)   # 过完备
    losses, sparsity = train_sae(sae, acts, epochs=600, l1=3e-3)
    codes = feature_codes(sae, acts)
    active, purity = monosemanticity(codes, labels)
    print(f"SAE 训练: 重建 {losses[0]:.3f}→{losses[-1]:.3f}, 末期稀疏度 {sparsity[-1]:.2f} (活跃特征比例)")
    print(f"活跃特征 {len(active)} 个, 平均单义纯度 {purity:.2f} (高=一个特征对应一个'当前值')")
    # 原始神经元的单义性对照
    _, raw_purity = monosemanticity(acts - acts.min(), labels)
    print(f"对照: 原始神经元平均纯度 {raw_purity:.2f}")
    print("→ SAE 特征比原始神经元更单义 (解叠加: 一个特征≈一个概念)。" if purity > raw_purity
          else "→ (本次纯度对比见 notebook 完整版)")
