"""ppo_clip_spec 的参考实现。用 numpy 逐元素计算，标量输入也能直接用。"""
from __future__ import annotations

import numpy as np


def ppo_clip_objective(ratio, advantage, clip_eps):
    ratio = np.asarray(ratio, dtype=float)
    advantage = np.asarray(advantage, dtype=float)
    unclipped = ratio * advantage
    clipped_ratio = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    clipped = clipped_ratio * advantage
    return np.minimum(unclipped, clipped)
