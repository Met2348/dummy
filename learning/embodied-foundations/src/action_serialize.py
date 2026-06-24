"""
action_serialize.py — 把观测/动作序列化成 token (tokens-as-actions 范式, M11.1).

为什么需要它 (M11.1): RT-2/OpenVLA 的核心范式是「**动作也当 token**」—— 把机器人动作离散成
token, 接到 VLM 的词表后面, 于是「控制机器人」= 「预测下一个 token」(复用你 LLM 的全套)。
本文件演示这个序列化: 连续 2D 动作 ↔ 离散动作 token。

离散化: 把 [dx,dy] 方向离散成 9 个动作 token (8 方向 + 停), 或 K×K 网格。这里用 8 方向+停 = 9 类。
"""
from __future__ import annotations

import numpy as np

# 9 个离散动作: 8 个方向 (45° 一档) + 停
_DIRS = []
for k in range(8):
    ang = k * np.pi / 4
    _DIRS.append([np.cos(ang), np.sin(ang)])
_DIRS.append([0.0, 0.0])                       # 停
ACTION_TOKENS = np.array(_DIRS, dtype=np.float32)   # (9, 2)
N_ACTION_TOKENS = len(ACTION_TOKENS)               # 9


def action_to_token(action: np.ndarray) -> int:
    """连续动作 → 最近的离散动作 token id (0..8)。"""
    a = np.asarray(action, dtype=np.float32)
    n = np.linalg.norm(a)
    if n < 1e-6:
        return N_ACTION_TOKENS - 1                 # 停
    a = a / n
    sims = ACTION_TOKENS[:-1] @ a                   # 与 8 个方向的余弦相似
    return int(np.argmax(sims))


def token_to_action(token: int) -> np.ndarray:
    """动作 token → 连续动作向量。"""
    return ACTION_TOKENS[int(token)].copy()


def serialize_episode(states: np.ndarray, actions: np.ndarray):
    """把一个 episode 的 (状态, 动作) 序列化: 状态当上下文向量, 动作离散成 token 序列。
    返回 (states (T,4), action_tokens (T,))。这是 tokens-as-actions 的最小形态。"""
    toks = np.array([action_to_token(a) for a in actions], dtype=np.int64)
    return states, toks


def roundtrip_error(actions: np.ndarray) -> float:
    """离散化损失: 连续动作 → token → 连续, 与原方向的平均余弦差 (越小越好)。"""
    errs = []
    for a in actions:
        n = np.linalg.norm(a)
        if n < 1e-6:
            continue
        rec = token_to_action(action_to_token(a))
        rn = np.linalg.norm(rec)
        if rn < 1e-6:
            continue
        cos = (a / n) @ (rec / rn)
        errs.append(1 - cos)
    return float(np.mean(errs)) if errs else 0.0


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import toy_env
    print(f"动作 token 表: {N_ACTION_TOKENS} 个 (8 方向 + 停)")
    S, A = toy_env.make_demos(n=50, seed=0)
    states, toks = serialize_episode(S, A)
    print(f"序列化: {S.shape[0]} 步 → 动作 token 序列, 样例: {toks[:12]}")
    print(f"离散化损失 (方向余弦差): {roundtrip_error(A):.3f} (8 方向, 应较小)")
