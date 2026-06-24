"""
video_diffusion.py — 视频扩散 (时空), 在玩具「运动序列」上看时序连贯性.

为什么需要它 (M13.4): 扩散从图 (M13.1-13.3) 扩到视频 = 加时间维。视频 = 一串帧, 关键是
**时序连贯 (temporal coherence)**: 帧之间要平滑过渡, 不能闪烁/跳变。核心难点就是「怎么让
扩散生成时间上连贯的序列」。

本文件用玩具「运动轨迹」当视频: 一段视频 = 一个 2D 点在 T 帧里的运动轨迹 (T×2)。数据分布是
**平滑轨迹** (点沿曲线移动)。对比:
  - 时空扩散 (joint): 模型看整条轨迹去噪 → 生成连贯 (平滑) 轨迹
  - 逐帧扩散 (per-frame): 每帧独立去噪 → 生成抖动 (不连贯) 轨迹
用「帧间跳变」量化时序连贯性 (越小越连贯)。

纯 torch (tiny CPU), 确定性。机制和真视频扩散一致, 只是「帧」是 2D 点而非图。
"""
from __future__ import annotations

import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

T_FRAMES = 12   # 每段视频的帧数


def make_trajectories(n: int = 400, T: int = T_FRAMES, seed: int = 0) -> np.ndarray:
    """合成「视频」数据集: n 条平滑 2D 运动轨迹 (n, T, 2). 每条是点沿一段正弦曲线移动。
    数据分布 = 平滑轨迹; 扩散要学会生成连贯 (平滑) 的运动。"""
    rng = np.random.default_rng(seed)
    ts = np.linspace(0, 1, T)
    trajs = []
    for _ in range(n):
        # 随机相位/频率/方向, 但每条都是平滑曲线
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(1.0, 2.0)
        amp = rng.uniform(0.5, 1.0)
        x = ts * 2 - 1                              # 横向匀速
        y = amp * np.sin(2 * np.pi * freq * ts + phase)  # 纵向正弦
        traj = np.stack([x, y], 1) + 0.02 * rng.standard_normal((T, 2))
        trajs.append(traj)
    return np.array(trajs, dtype=np.float32)        # (n, T, 2)


def make_beta_schedule(T: int = 50):
    betas = np.linspace(1e-4, 0.2, T).astype(np.float32)
    return betas, (1 - betas), np.cumprod(1 - betas).astype(np.float32)


def build_video_denoiser(T: int = T_FRAMES, d_model: int = 48, joint: bool = True, seed: int = 0):
    """视频去噪器. joint=True: 时空 transformer 看整条轨迹 (连贯); joint=False: 逐帧 MLP (不连贯)。"""
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        print(f"[video_diffusion] 无 torch ({exc!r})")
        return None
    torch.manual_seed(seed)

    class Joint(nn.Module):
        """时空: 把 T 帧当 token 序列, transformer 让帧之间互相 attend → 学时序结构。"""
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, d_model)
            self.pos = nn.Parameter(torch.randn(1, T, d_model) * 0.02)
            self.t_embed = nn.Linear(1, d_model)
            self.enc = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, 4, d_model * 2, batch_first=True), 2)
            self.out = nn.Linear(d_model, 2)

        def forward(self, x, t):  # x:(B,T,2), t:(B,1)
            h = self.proj(x) + self.pos + self.t_embed(t).unsqueeze(1)
            return self.out(self.enc(h))

    class PerFrame(nn.Module):
        """逐帧: 每帧独立去噪 (不看其它帧) → 帧间不连贯 (抖动)。"""
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2 + 1, d_model), nn.SiLU(),
                                     nn.Linear(d_model, d_model), nn.SiLU(),
                                     nn.Linear(d_model, 2))

        def forward(self, x, t):  # x:(B,T,2)
            import torch
            B, T_, _ = x.shape
            tt = t.unsqueeze(1).expand(B, T_, 1)
            return self.net(torch.cat([x, tt], -1))   # 每帧独立, 无跨帧交互

    return Joint() if joint else PerFrame()


def train_video(model, data: np.ndarray, T_diff: int = 50, epochs: int = 500,
                lr: float = 2e-3, seed: int = 0):
    """训练视频去噪器 (预测噪声). 返回 (loss 历史, schedule)。"""
    import torch
    import torch.nn as nn
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    betas, alphas, abars = make_beta_schedule(T_diff)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    losses = []
    for _ in range(epochs):
        t = rng.integers(0, T_diff, size=len(data))
        ab = abars[t][:, None, None]
        eps = rng.standard_normal(data.shape).astype(np.float32)
        xt = (np.sqrt(ab) * data + np.sqrt(1 - ab) * eps).astype(np.float32)
        pred = model(torch.tensor(xt), torch.tensor((t[:, None] / T_diff).astype(np.float32)))
        loss = lossf(pred, torch.tensor(eps))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses, (betas, alphas, abars)


def sample_videos(model, schedule, n: int = 100, T: int = T_FRAMES, seed: int = 0):
    """生成 n 条视频轨迹 (反向去噪)。"""
    import torch
    betas, alphas, abars = schedule
    T_diff = len(betas)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, T, 2)).astype(np.float32)
    with torch.no_grad():
        for t in reversed(range(T_diff)):
            tn = torch.full((n, 1), t / T_diff)
            eps = model(torch.tensor(x), tn).numpy()
            ab = abars[t]; a = alphas[t]; b = betas[t]
            mean = (1 / np.sqrt(a)) * (x - (b / np.sqrt(1 - ab)) * eps)
            x = mean + (np.sqrt(b) * rng.standard_normal((n, T, 2)).astype(np.float32) if t > 0 else 0)
    return x


def temporal_coherence(videos: np.ndarray) -> float:
    """时序连贯性 = 平均帧间跳变 (相邻帧距离). 越小越连贯 (平滑)。"""
    diffs = np.linalg.norm(videos[:, 1:] - videos[:, :-1], axis=-1)
    return float(diffs.mean())


if __name__ == "__main__":
    data = make_trajectories(n=400, seed=1)
    print(f"视频数据: {data.shape} (400 条轨迹, 每条 {T_FRAMES} 帧)")
    real_coh = temporal_coherence(data)
    print(f"真实轨迹的帧间跳变 (连贯基准): {real_coh:.3f}")
    for joint in [True, False]:
        model = build_video_denoiser(joint=joint)
        if model is None:
            break
        losses, sched = train_video(model, data, epochs=500)
        gen = sample_videos(model, sched, n=100, seed=2)
        coh = temporal_coherence(gen)
        tag = "时空 (joint)" if joint else "逐帧 (per-frame)"
        print(f"  {tag:18}: loss→{losses[-1]:.3f}, 生成帧间跳变 {coh:.3f} "
              f"({'连贯 ✅' if coh < real_coh*1.8 else '抖动 ⚠'})")
    print("→ 时空模型生成连贯轨迹; 逐帧模型抖动 (时序连贯是视频扩散核心难点, M13.4)。")
