"""
contrastive.py — 对比学习的核心 (CLIP/SigLIP 的灵魂): 把图和文拉进同一个空间.

为什么需要它: CLIP 让「一张猫的图」和「文本 a cat」在向量空间里靠近, 让不匹配的远离。
这个「图文对齐」是几乎所有 VLM 的视觉塔 (SigLIP/CLIP) 的训练方式。理解 InfoNCE / sigmoid
两种对比损失, 你就理解了视觉表示为什么「懂语言」。

本文件给:
  - info_nce_loss   : CLIP 的对称对比损失 (softmax over batch)
  - sigmoid_loss    : SigLIP 的成对 sigmoid 损失 (不需要全 batch 归一化, 更稳)
  - similarity_matrix: 图文相似度矩阵 (对角线应最亮 = 配对成功)

纯 numpy (+ 可选 torch 训练). 用合成「图嵌入/文嵌入」演示, 离线确定性。
"""
from __future__ import annotations

import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def similarity_matrix(img_emb: np.ndarray, txt_emb: np.ndarray) -> np.ndarray:
    """图文余弦相似度矩阵 S[i,j] = sim(图i, 文j). 配对正确时对角线最大。"""
    return l2_normalize(img_emb) @ l2_normalize(txt_emb).T


def info_nce_loss(img_emb: np.ndarray, txt_emb: np.ndarray, temp: float = 0.07) -> float:
    """CLIP 的对称 InfoNCE 损失 (numpy, 仅前向算损失值, 教学用).

    思想: 一个 batch 里, 第 i 张图的「正样本」是第 i 段文 (配对的), 其余 B-1 段文是负样本。
    损失 = 让 S[i,i] 在第 i 行 (图→文) 和第 i 列 (文→图) 的 softmax 里都最大。
    逐项:
      S    = 图文相似度矩阵 / temp   (temp 温度: 越小越「尖锐」, 放大正负差距)
      行方向 softmax 的交叉熵 (图 i 找对的文) + 列方向 (文 i 找对的图), 取平均。
    """
    S = similarity_matrix(img_emb, txt_emb) / temp
    B = S.shape[0]
    labels = np.arange(B)

    def ce_rows(M):  # 对每行做 softmax 交叉熵, 目标是对角
        M = M - M.max(axis=1, keepdims=True)
        logp = M - np.log(np.exp(M).sum(axis=1, keepdims=True))
        return -logp[labels, labels].mean()

    return float((ce_rows(S) + ce_rows(S.T)) / 2)


def sigmoid_loss(img_emb: np.ndarray, txt_emb: np.ndarray, temp: float = 0.07,
                 bias: float = 0.0) -> float:
    """SigLIP 的成对 sigmoid 损失 (numpy, 教学).

    和 InfoNCE 的关键区别: 不在整个 batch 上做 softmax 归一化, 而是把每个 (图i,文j) 对
    当独立的二分类 (配对=1 / 不配对=-1) 做 sigmoid。好处: 不需要大 batch、数值更稳、可扩展。
    逐项: z = S/temp + bias; 正对 (i==j) 标签 +1, 负对 -1; loss = -log sigmoid(label * z) 平均。
    """
    S = similarity_matrix(img_emb, txt_emb) / temp + bias
    B = S.shape[0]
    labels = -np.ones((B, B))
    np.fill_diagonal(labels, 1.0)
    # -log sigmoid(label * z) = softplus(-label * z)
    z = labels * S
    loss = np.logaddexp(0, -z)  # softplus(-z)
    return float(loss.mean())


def make_paired_embeddings(n: int = 6, dim: int = 8, noise: float = 0.3,
                           seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """造 n 对「配对的」图/文嵌入: 文嵌入 = 对应图嵌入 + 噪声 (模拟对齐但不完全相同)。
    噪声越小, 对角线越亮 (对齐越好)。用于演示对比损失。"""
    rng = np.random.default_rng(seed)
    img = rng.standard_normal((n, dim))
    txt = img + noise * rng.standard_normal((n, dim))   # 配对的文 ≈ 图 + 噪声
    return img, txt


if __name__ == "__main__":
    img, txt = make_paired_embeddings(n=6, noise=0.3, seed=1)
    S = similarity_matrix(img, txt)
    print("图文相似度矩阵 (对角线应最大):")
    print(np.round(S, 2))
    print(f"\n对角均值 {S.diagonal().mean():.3f} vs 非对角均值 "
          f"{(S.sum()-S.trace())/(S.size-len(S)):.3f}  (前者应明显更大)")
    print(f"InfoNCE 损失 (噪声0.3): {info_nce_loss(img, txt):.3f}")
    print(f"Sigmoid  损失 (噪声0.3): {sigmoid_loss(img, txt):.3f}")
    # 噪声更大 → 对齐更差 → 损失更高
    img2, txt2 = make_paired_embeddings(n=6, noise=1.5, seed=1)
    print(f"InfoNCE 损失 (噪声1.5, 对齐更差): {info_nce_loss(img2, txt2):.3f}  (应更高)")
