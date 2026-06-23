"""
tiny_vit.py — 一个能在 CPU 上秒跑的最小 Vision Transformer (ViT), 看清「图像如何变成 token」.

为什么从这里开始 (M10 地基): VLM 的视觉一半, 起点是「把图像切成 patch、当 token 喂 transformer」。
这个文件把 ViT 砍到最小可理解: patchify → 线性投影成 patch embedding → 加位置编码 → transformer
编码 → 得到一串视觉 token。理解了它, 你就理解了 CLIP/SigLIP/DINOv2/LLaVA 的视觉塔在做什么。

确定性 (固定 seed), 用合成图 (numpy 画的色块/条纹), 不依赖任何真实数据集, 离线可跑。
"""
from __future__ import annotations

import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def make_synthetic_image(kind: str = "blocks", size: int = 16, seed: int = 0) -> np.ndarray:
    """生成一张确定性合成图 (H,W,3), 取值 0-1. kind: blocks/stripes/gradient.
    用合成图是为了离线、可复现, 且 patch 结构清晰好可视化。"""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.float32)
    if kind == "blocks":
        h = size // 2
        for (r, c) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            color = rng.random(3)
            img[r * h:(r + 1) * h, c * h:(c + 1) * h] = color
    elif kind == "stripes":
        for i in range(size):
            img[i, :] = (i % 4) / 3.0
    else:  # gradient
        xs = np.linspace(0, 1, size)
        img[:] = xs[None, :, None]
    return img


def patchify(img: np.ndarray, patch: int = 4) -> np.ndarray:
    """把 (H,W,C) 图切成 (num_patches, patch*patch*C) 的扁平 patch 序列.
    这是 ViT 的第一步: 图 → 一串 patch (每个 patch 将成为一个 token)。"""
    H, W, C = img.shape
    assert H % patch == 0 and W % patch == 0
    nh, nw = H // patch, W // patch
    out = []
    for i in range(nh):
        for j in range(nw):
            blk = img[i * patch:(i + 1) * patch, j * patch:(j + 1) * patch, :]
            out.append(blk.reshape(-1))
    return np.stack(out)  # (nh*nw, patch*patch*C)


def build_tiny_vit(patch_dim: int, d_model: int = 32, n_heads: int = 4,
                   n_layers: int = 2, n_patches: int = 16, seed: int = 0):
    """构建最小 ViT (torch). 返回 (model, forward_fn). 失败 (无 torch) 时返回 None, 不报错."""
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        print(f"[tiny_vit] 无 torch ({exc!r}); 仅用 numpy 部分 (patchify) 演示。")
        return None

    torch.manual_seed(seed)

    class TinyViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(patch_dim, d_model)            # patch → embedding
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))  # CLS token (汇总全图)
            self.pos = nn.Parameter(torch.randn(1, n_patches + 1, d_model) * 0.02)
            layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2,
                                               batch_first=True)
            self.enc = nn.TransformerEncoder(layer, n_layers)

        def forward(self, patches):  # patches: (B, n_patches, patch_dim)
            B = patches.shape[0]
            x = self.proj(patches)                               # (B, N, d)
            cls = self.cls.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1) + self.pos            # 前置 CLS + 位置编码
            x = self.enc(x)
            return x                                             # (B, N+1, d); x[:,0] 是图级表示

    return TinyViT()


if __name__ == "__main__":
    img = make_synthetic_image("blocks", size=16, seed=1)
    patches = patchify(img, patch=4)
    print(f"图 {img.shape} → patchify → {patches.shape} (16 个 patch, 每个 {patches.shape[1]} 维)")
    model = build_tiny_vit(patch_dim=patches.shape[1], n_patches=patches.shape[0])
    if model is not None:
        import torch
        toks = model(torch.tensor(patches[None], dtype=torch.float32))
        print(f"ViT 输出视觉 token: {tuple(toks.shape)} (含 1 个 CLS + 16 个 patch token)")
        print("→ 这串 token 就是后面接进 LLM 的「视觉一半」(见 M10.2 vl-fusion)。")
