"""生成 13.4 notebooks (N1 时空 vs 逐帧视频扩散 / N2 时序连贯度量). 跑后 nbconvert --execute。"""
from __future__ import annotations
import sys
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
def md(s): return new_markdown_cell(s)
def code(s): return new_code_cell(s)
MPL = """import matplotlib, matplotlib.pyplot as plt
matplotlib.rcParams['axes.unicode_minus']=False
for f in ['Microsoft YaHei','SimHei','DejaVu Sans']:
    try: matplotlib.rcParams['font.sans-serif']=[f]; break
    except Exception: pass"""
PATHS = """import sys
from pathlib import Path
SRC = Path.cwd().parent / "src"
sys.path.insert(0, str(SRC))"""

# ── N1 video-diffusion ──
n1 = new_notebook()
n1.cells = [
    md("""# N1 · 时空 vs 逐帧视频扩散 (时序连贯从哪来)

> 配套 13.4-L1 · 用「运动轨迹」当视频 (一段视频 = 一个 2D 点在 T 帧的轨迹)。
> 训两个扩散模型: 时空 (joint, 帧互相 attend) vs 逐帧 (per-frame, 独立)。
> 亲手看到: **时序连贯来自联合建模时间, 不是免费的**。"""),
    code(PATHS + "\nimport numpy as np, torch\nimport video_diffusion as vd\nprint('video_diffusion 就绪, 每段视频', vd.T_FRAMES, '帧')"),
    md("## 1. 视频数据 = 平滑运动轨迹. 扩散要学会生成连贯的运动"),
    code(MPL + """
data = vd.make_trajectories(n=400, seed=1)
print('视频数据:', data.shape, '(400 条轨迹, 每条', vd.T_FRAMES, '帧, 每帧 2D)')
fig, ax = plt.subplots(figsize=(5,4))
for i in range(8):
    ax.plot(data[i,:,0], data[i,:,1], '-o', ms=3, alpha=0.6)
ax.set_title('真实「视频」: 平滑运动轨迹 (8 条样例)'); ax.set_xlabel('x'); ax.set_ylabel('y')
plt.tight_layout(); plt.show()
print(f'真实轨迹帧间跳变 (连贯基准): {vd.temporal_coherence(data):.3f}')"""),
    md("""## 2. 训练两个视频扩散模型
- **时空 (joint)**: transformer 把 T 帧当 token 序列, 帧之间互相 attend (L1 正解)
- **逐帧 (per-frame)**: MLP 每帧独立去噪, 不看其它帧 (L1 反例)"""),
    code("""torch.manual_seed(0)
models = {}
for joint in [True, False]:
    m = vd.build_video_denoiser(joint=joint, seed=0)
    losses, sched = vd.train_video(m, data, epochs=500, seed=0)
    models[joint] = (m, sched, losses)
    tag = '时空(joint)' if joint else '逐帧(per-frame)'
    print(f'{tag:18} 训练 loss {losses[0]:.3f} → {losses[-1]:.3f}')"""),
    md("## 3. 各生成 100 条轨迹, 看连贯性 (帧间跳变越小越连贯)"),
    code(MPL + """
real_coh = vd.temporal_coherence(data)
fig, axes = plt.subplots(1, 2, figsize=(11,4.2))
for ax, joint in zip(axes, [True, False]):
    m, sched, _ = models[joint]
    gen = vd.sample_videos(m, sched, n=100, seed=2)
    coh = vd.temporal_coherence(gen)
    for i in range(8):
        ax.plot(gen[i,:,0], gen[i,:,1], '-o', ms=3, alpha=0.6)
    tag = '时空 (joint)' if joint else '逐帧 (per-frame)'
    verdict = '连贯 ✅' if coh < real_coh*1.8 else '抖动 ⚠'
    ax.set_title(f'{tag}\\n帧间跳变 {coh:.3f} ({verdict})'); ax.set_xlabel('x'); ax.set_ylabel('y')
plt.suptitle(f'生成的「视频」轨迹 (真实基准 {real_coh:.3f})'); plt.tight_layout(); plt.show()
print('→ 时空模型生成平滑连贯轨迹; 逐帧模型抖动 (帧间乱跳)。')"""),
    md("""## 4. 反思
你亲手验证了视频扩散的核心: **时序连贯来自联合建模时间**。带走:
- 同样的 DDPM 框架 (M13.1), 数据从「2D 点」变成「T 帧轨迹」, 去噪网络从 MLP 变成时空 transformer。
- **逐帧独立 → 抖动; 时空联合 (帧互相 attend) → 连贯**。连贯不是免费的, 来自跨帧建模。
- 真实视频 (Sora) 是同一个东西的放大版: 把「2D 点」换成「时空 latent patch」(N1→L2)。
下一步 N2: 把「连贯」量化成度量, 并体会长程一致比帧间更难 (L3)。"""),
]
nbformat.write(n1, HERE / "N1-video-diffusion.ipynb")
print("written N1")

# ── N2 temporal-coherence ──
n2 = new_notebook()
n2.cells = [
    md("""# N2 · 时序连贯度量 (帧间 vs 长程)

> 配套 13.4-L3 · 把「连贯」量化成度量。看两个尺度:
> **帧间跳变** (易, 时空建模能搞定) vs **长程一致** (难, open 问题)。
> 体会 L3 的核心: 连贯是分尺度的, 越长越难。"""),
    code(PATHS + "\nimport numpy as np, torch\nimport video_diffusion as vd\nprint('就绪')"),
    md("## 1. 训练时空 vs 逐帧两个模型 (复用 N1)"),
    code("""torch.manual_seed(0)
data = vd.make_trajectories(n=400, seed=1)
gens = {}
for joint in [True, False]:
    m = vd.build_video_denoiser(joint=joint, seed=0)
    _, sched = vd.train_video(m, data, epochs=500, seed=0)
    gens[joint] = vd.sample_videos(m, sched, n=200, seed=3)
print('生成完毕: 时空', gens[True].shape, '逐帧', gens[False].shape)"""),
    md("""## 2. 两个尺度的连贯度量
- **帧间跳变** = 相邻帧平均距离 (L1/L2 能搞定的尺度)
- **长程代理** = 首尾段「速度方向」一致性 (L3 难尺度的玩具代理)"""),
    code("""def frame_jitter(v):           # 帧间: 相邻帧距离
    return float(np.linalg.norm(v[:,1:]-v[:,:-1], axis=-1).mean())

def longrange_drift(v):        # 长程代理: 整体运动方向的稳定度 (首尾速度方向夹角)
    vel = v[:,1:]-v[:,:-1]                       # (n,T-1,2) 逐帧速度
    early = vel[:,:3].mean(1); late = vel[:,-3:].mean(1)  # 头/尾平均速度
    cos = (early*late).sum(-1) / (np.linalg.norm(early,axis=-1)*np.linalg.norm(late,axis=-1)+1e-8)
    return float(np.clip(cos,-1,1).mean())       # 越接近真实越好

real_j, real_l = frame_jitter(data), longrange_drift(data)
print(f'真实:   帧间跳变 {real_j:.3f}   长程方向一致 {real_l:+.3f}')
for joint in [True, False]:
    tag = '时空' if joint else '逐帧'
    print(f'{tag}: 帧间跳变 {frame_jitter(gens[joint]):.3f}   长程方向一致 {longrange_drift(gens[joint]):+.3f}')"""),
    md("## 3. 可视化两个尺度的差距"),
    code(MPL + """
labels = ['真实', '时空(joint)', '逐帧(per-frame)']
jit = [real_j, frame_jitter(gens[True]), frame_jitter(gens[False])]
lr  = [real_l, longrange_drift(gens[True]), longrange_drift(gens[False])]
fig, axes = plt.subplots(1, 2, figsize=(11,4))
axes[0].bar(labels, jit, color=['gray','C0','C3']); axes[0].set_title('帧间跳变 (越小越连贯, 易尺度)')
axes[0].axhline(real_j, ls='--', c='gray', alpha=0.6)
axes[1].bar(labels, lr, color=['gray','C0','C3']); axes[1].set_title('长程方向一致 (越接近真实越好, 难尺度)')
axes[1].axhline(real_l, ls='--', c='gray', alpha=0.6)
plt.suptitle('连贯是分尺度的: 帧间 (易) vs 长程 (难)'); plt.tight_layout(); plt.show()"""),
    md("""## 4. 反思 (13.4 收口)

你把「连贯」量化成了度量, 并看到它**分尺度**。带走:
- **帧间跳变**: 时空建模就能压到接近真实 (L1/L2 的胜利)。
- **长程一致**: 即使帧间好, 长程方向/物体身份仍可能漂 —— 这是 open 难题 (L3), 接 M13.5 世界模型 (需持久状态记住远处) 和 M11 具身 (物理一致)。
- 评估是瓶颈: 没有好指标抓「物理合理/长程一致」, 还得人/VLM 评 (接 M10.6 + 评估模块)。

> **M13.4 收口**: 视频 = 图扩散 + 时间维; 连贯来自时空联合建模; Sora = 你学过的部件 + 规模; 成本驱动架构。
> **交棒 M13.5「world-models」**: 视频生成再进一步 —— 能预测「动作后世界变成什么」就是世界模型 (可交互视频 + 长程一致 + 接 M11 具身规划)。下一专题 `world-models`。"""),
]
nbformat.write(n2, HERE / "N2-temporal-coherence.ipynb")
print("written N2")
