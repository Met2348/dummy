"""
temporal_tokens.py — 把视频 (图像 + 时间) 变成时空 token, 把 M10.4 的 token 化扩到时序.

为什么需要它 (M10.5): M10.4 把单图变 token。视频 = 一串图 + 时间维。怎么 token 化? 朴素做法
是「每帧独立 patch」(忽略时间), 但好做法是**时空 patch (spatiotemporal)**: 把「几帧 × 一块
空间」当一个时空立方体, 切成一个 token —— 既编码空间、又编码运动。这是 ViViT / Sora 式视频
模型的核心。

本文件给:
  - make_toy_video    : 合成「移动色块」视频 (T,H,W,3), 确定性, 看得见运动
  - spatial_patchify  : 朴素逐帧 patch (忽略时间, 作对照)
  - spatiotemporal_patchify : 时空 patch (帧×空间立方体), 编码运动
  - token 数对比: 时空 patch 如何压缩视频 token 数

纯 numpy, 离线确定性, 无需真实视频。
"""
from __future__ import annotations

import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def make_toy_video(T: int = 8, size: int = 8, seed: int = 0) -> np.ndarray:
    """合成一段「移动色块」视频 (T, H, W, 3). 一个色块从左上移到右下, 能看见运动。"""
    rng = np.random.default_rng(seed)
    color = rng.random(3)
    vid = np.zeros((T, size, size, 3), dtype=np.float32)
    block = max(2, size // 4)
    for t in range(T):
        # 色块位置随时间线性移动
        pos = int(t / max(T - 1, 1) * (size - block))
        vid[t, pos:pos + block, pos:pos + block] = color
    return vid


def spatial_patchify(vid: np.ndarray, patch: int = 2) -> np.ndarray:
    """朴素: 每帧独立切空间 patch (忽略时间). 返回 (T * n_spatial, patch*patch*3)。
    token 数 = 帧数 × 每帧 patch 数 (随帧线性增长, 容易爆)。"""
    T, H, W, C = vid.shape
    out = []
    for t in range(T):
        for i in range(H // patch):
            for j in range(W // patch):
                out.append(vid[t, i*patch:(i+1)*patch, j*patch:(j+1)*patch].reshape(-1))
    return np.stack(out)


def spatiotemporal_patchify(vid: np.ndarray, patch: int = 2, tpatch: int = 2) -> np.ndarray:
    """时空 patch: 把 (tpatch 帧 × patch×patch 空间) 当一个时空立方体, 切成一个 token.
    返回 (n_tokens, tpatch*patch*patch*3). token 数 = (T/tpatch) × n_spatial (比朴素少 tpatch 倍)。
    每个 token 编码了「一小段时间里一小块空间」—— 含运动信息。"""
    T, H, W, C = vid.shape
    assert T % tpatch == 0
    out = []
    for ti in range(T // tpatch):
        for i in range(H // patch):
            for j in range(W // patch):
                cube = vid[ti*tpatch:(ti+1)*tpatch,
                           i*patch:(i+1)*patch, j*patch:(j+1)*patch, :]
                out.append(cube.reshape(-1))
    return np.stack(out)


def token_count_comparison(vid: np.ndarray, patch: int = 2, tpatch: int = 2) -> dict:
    """对比朴素逐帧 vs 时空 patch 的 token 数 (时空压缩了多少)。"""
    sp = spatial_patchify(vid, patch).shape[0]
    st = spatiotemporal_patchify(vid, patch, tpatch).shape[0]
    return {"逐帧 patch token 数": sp, "时空 patch token 数": st,
            "压缩比": round(sp / st, 1), "时间 patch 大小": tpatch}


def motion_signal(vid: np.ndarray) -> np.ndarray:
    """相邻帧差的能量 (量化「有多少运动」). 时空 token 必须能编码这个。"""
    return np.array([np.abs(vid[t] - vid[t-1]).mean() for t in range(1, len(vid))])


if __name__ == "__main__":
    vid = make_toy_video(T=8, size=8, seed=1)
    print(f"合成视频: {vid.shape} (8 帧, 8x8, 3 通道)")
    print(f"逐帧运动能量: {np.round(motion_signal(vid), 3).tolist()}  (色块在动)")
    cmp = token_count_comparison(vid, patch=2, tpatch=2)
    for k, v in cmp.items():
        print(f"  {k}: {v}")
    print("→ 时空 patch 把 token 数压缩了, 还编码了运动 (一个 token = 一小段时空)。")
