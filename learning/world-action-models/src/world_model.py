"""
world_model.py — 具身世界模型 + 规划 (M11.5). 与 M13.5 同源 (世界模型概念), 这里用于机器人控制.

为什么需要它 (M11.5): 从「纯模仿」(M11.4 BC) 到「带想象」。世界模型 = 学环境转移
(state, action → next state)。学会后能在脑内**想象 rollout** 并**规划** (MPC): 不靠专家、只靠
**随机探索**数据学动态, 再规划出到目标的动作。这是 model-based 机器人 (NVIDIA「先学想象, 再学
行动」路线), 也接你 M13.5 的 world_model.py (同一概念, M13.5 是扩散侧的孪生)。

复用 M11.1 toy_env (2D 到达)。关键对比:
  - model-based (本专题): 从**随机**转移学世界模型 → MPC 规划 → 到目标 (无需专家!)
  - model-free (M11.4 BC):  需**专家** demo 才能学
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

_M11 = Path(__file__).resolve().parents[2] / "embodied-foundations" / "src"
if str(_M11) not in sys.path:
    sys.path.insert(0, str(_M11))
import toy_env as env  # noqa: E402


def make_random_transitions(n: int = 4000, seed: int = 0):
    """从**随机**探索采转移 (state, action, agent_delta), 不需要专家! 这是 model-based 的数据优势。"""
    rng = np.random.default_rng(seed)
    S, A, D = [], [], []
    for _ in range(n):
        s = env.reset(rng)
        a = rng.uniform(-1, 1, size=env.ACT_DIM).astype(np.float32)   # 随机动作
        ns = env.step(s, a)
        S.append(s); A.append(a); D.append((ns[:2] - s[:2]))          # 只学 agent 位移
    return np.array(S, np.float32), np.array(A, np.float32), np.array(D, np.float32)


def build_world_model(hidden: int = 64, seed: int = 0):
    """世界模型: (state, action) → agent 位移 Δ。"""
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        print(f"[world_model] 无 torch ({exc!r})"); return None
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(env.STATE_DIM + env.ACT_DIM, hidden), nn.SiLU(),
                         nn.Linear(hidden, hidden), nn.SiLU(),
                         nn.Linear(hidden, env.ACT_DIM))


def train_world_model(model, S, A, D, epochs: int = 400, lr: float = 3e-3, seed: int = 0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    X = torch.tensor(np.concatenate([S, A], 1)); Y = torch.tensor(D)
    opt = torch.optim.Adam(model.parameters(), lr=lr); lossf = nn.MSELoss()
    losses = []
    for _ in range(epochs):
        loss = lossf(model(X), Y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses


def imagine_next(model, state, action):
    """用世界模型想象下一状态 (不碰真环境)。"""
    import torch
    x = torch.tensor(np.concatenate([state, action])[None].astype(np.float32))
    delta = model(x).detach().numpy()[0]
    ns = state.copy()
    ns[:2] = np.clip(state[:2] + delta, -env.BOX, env.BOX)
    return ns


def plan_action(model, state, n_samples: int = 200, horizon: int = 6, seed: int = 0):
    """MPC (随机射击): 采 n_samples 条动作序列, 用世界模型想象 rollout, 选末端最接近目标的, 返回其首动作。"""
    rng = np.random.default_rng(seed)
    goal = state[2:]
    best_a, best_d = np.zeros(env.ACT_DIM, np.float32), 1e9
    for _ in range(n_samples):
        seq = rng.uniform(-1, 1, size=(horizon, env.ACT_DIM)).astype(np.float32)
        s = state.copy()
        for a in seq:
            s = imagine_next(model, s, a)
        d = np.linalg.norm(s[:2] - goal)
        if d < best_d:
            best_d, best_a = d, seq[0]
    return best_a


def mpc_policy_fn(model, n_samples: int = 200, horizon: int = 6):
    counter = {"n": 0}

    def fn(state):
        counter["n"] += 1
        return plan_action(model, state, n_samples=n_samples, horizon=horizon, seed=counter["n"])
    return fn


if __name__ == "__main__":
    print("具身世界模型 + MPC 规划 (model-based), 环境 = M11.1 2D 到达\n")
    S, A, D = make_random_transitions(n=4000, seed=0)
    print(f"随机探索转移 (无专家!): {len(S)} 个")
    model = build_world_model()
    losses = train_world_model(model, S, A, D, epochs=400)
    print(f"世界模型训练: loss {losses[0]:.4f} → {losses[-1]:.4f}")
    sr = env.eval_policy(mpc_policy_fn(model, n_samples=200, horizon=6), n_episodes=100)
    print(f"MPC 规划成功率: {sr:.2f} (只用随机数据学世界模型 + 规划, 没用任何专家 demo!)")
    print("→ model-based: 随机探索学动态 + 规划 → 解任务; 不需要专家 (vs M11.4 BC 需专家)。")
