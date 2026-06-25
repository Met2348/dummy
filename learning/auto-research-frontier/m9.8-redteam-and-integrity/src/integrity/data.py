"""两个数据集变体 + 指纹。用于演示"偷换数据集"攻击与"数据集指纹"守卫。

- easy-v1：干净可分，真值规则一打一个准（高准确率）。
- hard-v2：带噪声，同样规则只能到 ~0.85（这才是该报告的真实难度）。
偷换数据集 = 在 easy 上跑、却声称在 hard 上跑，骗一个虚高准确率。指纹（sha256）能戳穿它。
"""
from __future__ import annotations

import hashlib

import numpy as np


def make_dataset(name: str):
    """确定性生成 (X, y)。easy-v1 无噪声、hard-v2 有噪声。"""
    if name == "easy-v1":
        rng, noise = np.random.default_rng(1), 0.0
    elif name == "hard-v2":
        rng, noise = np.random.default_rng(2), 0.8
    else:
        raise KeyError(name)
    X = rng.normal(size=(120, 2))
    logits = X[:, 0] + X[:, 1] + rng.normal(scale=noise, size=120)
    y = (logits > 0).astype(int)
    return X, y


def fingerprint(X, y) -> str:
    """数据集指纹：内容的 sha256。换了数据，指纹必变。"""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()[:16]


# 已知数据集的"官方指纹"——守卫用它核对"声称的数据集"是否名副其实
KNOWN_FINGERPRINTS = {
    name: fingerprint(*make_dataset(name)) for name in ("easy-v1", "hard-v2")
}


def rule_predict(X, threshold: float = 0.0):
    """被研究的"模型"：predict 1 if x+y > threshold（真值边界是 0）。"""
    return (X[:, 0] + X[:, 1] > threshold).astype(int)


def accuracy(preds, labels) -> float:
    preds, labels = np.asarray(preds), np.asarray(labels)
    return float((preds == labels).mean())
