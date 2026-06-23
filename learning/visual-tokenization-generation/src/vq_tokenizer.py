"""
vq_tokenizer.py — 把图像变成「离散视觉 token」(VQ), 这是「让 LLM 画图」的关键.

为什么需要它 (M10.4 的核心): M10.3 的 VLM 会「读」图 (图当输入)。要让模型「画」图, 得把图变成
**离散 token** —— 像文本的词表一样。这样「生成图」就变成「自回归生成视觉 token」, 和生成文本
机制完全相同。VQ-VAE/VQGAN 就干这个: 学一个「视觉码本 (codebook)」, 把每块图量化成码本里
最近的那个条目的编号 (一个离散 token)。

本文件用最小、确定性的方式实现 VQ: 在图像 patch 上做 k-means 学码本, 把每个 patch 量化成
码本编号 (token), 再用码本重建。码本越大, 重建越精细 —— 这是 VQ 的核心权衡, 一眼可见。

纯 numpy (k-means 自己写), 离线确定性, 合成图。
"""
from __future__ import annotations

import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def image_to_patches(img: np.ndarray, patch: int = 2) -> np.ndarray:
    """(H,W,C) → (n_patch, patch*patch*C). 每个 patch 将被量化成一个离散 token。"""
    H, W, C = img.shape
    out = []
    for i in range(H // patch):
        for j in range(W // patch):
            out.append(img[i*patch:(i+1)*patch, j*patch:(j+1)*patch].reshape(-1))
    return np.stack(out)


def patches_to_image(patches: np.ndarray, grid: int, patch: int, C: int = 3) -> np.ndarray:
    """(n_patch, patch*patch*C) → (H,W,C). image_to_patches 的逆。"""
    H = W = grid * patch
    img = np.zeros((H, W, C))
    k = 0
    for i in range(grid):
        for j in range(grid):
            img[i*patch:(i+1)*patch, j*patch:(j+1)*patch] = patches[k].reshape(patch, patch, C)
            k += 1
    return img


def train_codebook(patches: np.ndarray, codebook_size: int = 16, iters: int = 30,
                   seed: int = 0) -> np.ndarray:
    """用 k-means 学一个视觉码本 (codebook_size 个原型). 返回 (codebook_size, patch_dim)。
    码本 = 视觉「词表」; 每个条目是一个常见的 patch 模式。"""
    rng = np.random.default_rng(seed)
    # k-means++ 式初始化 (取数据点当初始中心)
    idx = rng.choice(len(patches), size=min(codebook_size, len(patches)), replace=False)
    codebook = patches[idx].copy()
    if len(codebook) < codebook_size:  # patch 不够则补随机
        extra = rng.standard_normal((codebook_size - len(codebook), patches.shape[1])) * 0.1
        codebook = np.vstack([codebook, extra])
    for _ in range(iters):
        # 分配: 每个 patch 找最近码本条目
        d = ((patches[:, None, :] - codebook[None, :, :]) ** 2).sum(-1)
        assign = d.argmin(1)
        # 更新: 每个码本条目 = 分配给它的 patch 均值
        for c in range(codebook_size):
            members = patches[assign == c]
            if len(members) > 0:
                codebook[c] = members.mean(0)
    return codebook


def quantize(patches: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """把每个 patch 量化成最近码本条目的编号 (离散 token). 返回 (n_patch,) int token 序列。"""
    d = ((patches[:, None, :] - codebook[None, :, :]) ** 2).sum(-1)
    return d.argmin(1)


def dequantize(tokens: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """token 序列 → patch (查码本). 量化的逆。"""
    return codebook[tokens]


def reconstruct(img: np.ndarray, codebook_size: int = 16, patch: int = 2,
                seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """完整 VQ 流程: 图 → patch → 量化成 token → 反量化 → 重建图.
    返回 (重建图, token 序列, 重建 MSE)。"""
    patches = image_to_patches(img, patch)
    grid = img.shape[0] // patch
    codebook = train_codebook(patches, codebook_size, seed=seed)
    tokens = quantize(patches, codebook)
    recon_patches = dequantize(tokens, codebook)
    recon = patches_to_image(recon_patches, grid, patch, img.shape[2])
    mse = float(((img - recon) ** 2).mean())
    return recon, tokens, mse


def make_image(size: int = 8, seed: int = 0) -> np.ndarray:
    """一张确定性合成图 (彩色色块), 用于演示 VQ 重建。"""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3))
    h = size // 2
    for (r, c) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        img[r*h:(r+1)*h, c*h:(c+1)*h] = rng.random(3)
    return img + 0.05 * rng.standard_normal((size, size, 3))


if __name__ == "__main__":
    img = make_image(size=8, seed=1)
    print("码本大小 → 重建 MSE (越大码本, 重建越精细):")
    for k in [2, 4, 8, 16, 32]:
        recon, tokens, mse = reconstruct(img, codebook_size=k, patch=2, seed=1)
        print(f"  码本 {k:>2}: MSE {mse:.4f}, token 序列 = {tokens.tolist()}")
    print("\n→ 图被压成一串离散 token (像文本的词)。生成图 = 自回归生成这串 token (M10.4-L2)。")
