"""
mini_vla.py — 最小 VLA 架构: 感知 backbone + 可换动作头 (M11.2).

为什么需要它 (M11.2): VLA 的标准结构是**两阶段**: ① 感知/推理 backbone (真实里 = VLM, 你的 M10)
② 动作头 (把 backbone 的特征解码成动作)。本文件用 M11.1 的 toy_env 搭一个最小 VLA: 一个小
backbone 编码状态, 后面接**可换的动作头** (离散 token 头 vs 连续回归头), 看两者的代价。

  observation (state) → [backbone 编码] → feature → [action head] → action
                                                      ↑ 可换: 离散(9类) / 连续(2D)

真实 VLA 把 backbone 换成 M10 的 VLM (吃图像+指令), 动作头换成扩散/flow (M11.3)。结构一样, 规模不同。
复用 M11.1 的 toy_env (跨专题 src 复用)。纯 torch tiny CPU 确定性。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 复用 M11.1 的共享环境与动作序列化
_M11 = Path(__file__).resolve().parents[2] / "embodied-foundations" / "src"
if str(_M11) not in sys.path:
    sys.path.insert(0, str(_M11))
import action_serialize as ser  # noqa: E402
import toy_env as env  # noqa: E402


def build_mini_vla(head: str = "discrete", d_model: int = 64, seed: int = 0):
    """两阶段 VLA: backbone (状态编码) + 动作头 (head='discrete' 9类 / 'continuous' 2D)。"""
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        print(f"[mini_vla] 无 torch ({exc!r})"); return None
    torch.manual_seed(seed)

    class MiniVLA(nn.Module):
        def __init__(self):
            super().__init__()
            # backbone: 真实 VLA 这里是 VLM (M10); 玩具里是小 MLP 编码状态
            self.backbone = nn.Sequential(
                nn.Linear(env.STATE_DIM, d_model), nn.SiLU(),
                nn.Linear(d_model, d_model), nn.SiLU())
            self.head_type = head
            if head == "discrete":
                self.head = nn.Linear(d_model, ser.N_ACTION_TOKENS)   # 9 个动作 token logits
            elif head == "continuous":
                self.head = nn.Linear(d_model, env.ACT_DIM)           # 直接回归 2D 动作
            else:
                raise ValueError(head)

        def forward(self, s):
            return self.head(self.backbone(s))

    return MiniVLA()


def train_vla(model, S: np.ndarray, A: np.ndarray, epochs: int = 400, lr: float = 3e-3, seed: int = 0):
    """训练 mini-VLA (模仿专家)。discrete: CE on 动作 token; continuous: MSE on 动作。"""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    X = torch.tensor(S)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    if model.head_type == "discrete":
        Y = torch.tensor(np.array([ser.action_to_token(a) for a in A], dtype=np.int64))
        lossf = lambda out: nn.functional.cross_entropy(out, Y)
    else:
        Y = torch.tensor(A)
        lossf = lambda out: nn.functional.mse_loss(out, Y)
    for _ in range(epochs):
        loss = lossf(model(X))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses


def make_policy(model):
    """把 mini-VLA 包成 policy_fn(state)->action, 供 toy_env.rollout/eval。"""
    import torch

    def policy_fn(state):
        with torch.no_grad():
            out = model(torch.tensor(state[None], dtype=torch.float32))
        if model.head_type == "discrete":
            return ser.token_to_action(int(out.argmax(-1)))
        return out.numpy()[0]
    return policy_fn


def action_smoothness(model, seeds=range(30)) -> float:
    """动作平滑度 = rollout 中相邻动作的平均变化幅度 (越小越平滑)。"""
    pol = make_policy(model)
    changes = []
    for sd in seeds:
        _, _, traj = env.rollout(pol, seed=int(sd), record=True)
        if traj is None or len(traj) < 3:
            continue
        acts = [pol(s) for s in traj[:-1]]
        for i in range(1, len(acts)):
            changes.append(np.linalg.norm(np.array(acts[i]) - np.array(acts[i - 1])))
    return float(np.mean(changes)) if changes else 0.0


if __name__ == "__main__":
    S, A = env.make_demos(n=400, seed=0)
    for head in ["discrete", "continuous"]:
        m = build_mini_vla(head=head)
        if m is None:
            break
        losses = train_vla(m, S, A, epochs=400)
        sr = env.eval_policy(make_policy(m), n_episodes=200)
        sm = action_smoothness(m)
        print(f"{head:11} 头: loss→{losses[-1]:.3f}, 成功率 {sr:.2f}, 动作平滑度 {sm:.3f}")
    print("→ 离散头简单但动作'跳'(8方向量化); 连续头精度高、动作平滑。VLA = backbone + 可换动作头。")
