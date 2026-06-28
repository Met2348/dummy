"""生成 11.5 notebooks (N1 世界模型+MPC规划 / N2 model-based vs model-free 样本效率). 跑后 nbconvert --execute。"""
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
sys.path.insert(0, str(Path.cwd().parent / "src"))
import world_model as wm
import toy_env as env   # M11.1 共享 (world_model 已加进 path)
import numpy as np, torch"""

# ── N1 world-model-planning ──
n1 = new_notebook()
n1.cells = [
    md("""# N1 · 世界模型 + MPC 规划 (零专家!)

> 配套 11.5-L1 · 从**纯随机探索**学一个世界模型 (state,action→Δ), 然后用 MPC 规划到目标。
> 关键: **不用任何专家 demo** —— 这是 model-based 的数据优势 (vs M11.4 BC 需专家)。"""),
    code(PATHS + "\nprint('环境 = M11.1 2D 到达; 世界模型与 M13.5 同源 (一面生成一面决策)')"),
    md("## 1. 从随机探索学世界模型 (无专家)"),
    code("""torch.manual_seed(0)
S, A, D = wm.make_random_transitions(n=4000, seed=0)   # 随机动作 + 看结果, 零专家!
print(f'随机探索转移: {len(S)} 个 (state, action, Δ) — 没有任何专家动作')
model = wm.build_world_model()
losses = wm.train_world_model(model, S, A, D, epochs=400)
print(f'世界模型训练: loss {losses[0]:.4f} → {losses[-1]:.4f} (学会了环境怎么动)')"""),
    md("## 2. MPC 规划: 用世界模型想象 + 搜动作 → 到目标"),
    code(MPL + """
pol = wm.mpc_policy_fn(model, n_samples=200, horizon=6)
sr = env.eval_policy(pol, n_episodes=100)
print(f'MPC 规划成功率: {sr:.2f} (只用随机数据学世界模型 + 规划)')
fig, axes = plt.subplots(1,3,figsize=(13,4.2))
for ax, seed in zip(axes, [4,8,12]):
    ok,steps,traj = env.rollout(pol, seed=seed, record=True); traj=np.array(traj)
    ax.plot(traj[:,0],traj[:,1],'-o',ms=4); ax.plot(traj[0,0],traj[0,1],'gs',ms=12); ax.plot(traj[0,2],traj[0,3],'r*',ms=16)
    ax.add_patch(plt.Rectangle((-1,-1),2,2,fill=False,ls=':',ec='gray'))
    ax.set_title(f'MPC 规划 (成功={ok}, {steps}步)'); ax.set_aspect('equal')
plt.suptitle('世界模型 + MPC: 想象 rollout 搜动作 → 到目标 (零专家 demo!)'); plt.tight_layout(); plt.show()
print('→ 不学策略, 靠世界模型想象 + 搜索就能控制 (规划)。数据来自随机探索, 没用专家。')"""),
    md("## 3. 想象 vs 真实: 世界模型学得准吗 (误差累积)"),
    code(MPL + """
# 给一串动作, 对比世界模型想象 vs 真环境
rng = np.random.default_rng(3)
s0 = env.reset(rng)
acts = rng.uniform(-1,1,size=(10,2)).astype(np.float32)
imag = [s0.copy()]; real = [s0.copy()]
si, sr_ = s0.copy(), s0.copy()
for a in acts:
    si = wm.imagine_next(model, si, a); imag.append(si.copy())
    sr_ = env.step(sr_, a); real.append(sr_.copy())
imag=np.array(imag); real=np.array(real)
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(real[:,0],real[:,1],'-o',label='真环境',color='gray',ms=6)
ax.plot(imag[:,0],imag[:,1],'--s',label='世界模型想象',color='C0',ms=5)
ax.legend(); ax.set_title('想象紧贴真实 (误差随步数缓慢累积, M13.5)'); ax.set_aspect('equal')
plt.tight_layout(); plt.show()
errs=np.linalg.norm(imag[:,:2]-real[:,:2],axis=1)
print(f'想象误差: 1步 {errs[1]:.3f} → 10步 {errs[10]:.3f} (累积, 故 MPC 用短视野+重规划)')"""),
    md("""## 4. 反思
你用世界模型 + MPC 解了任务, **零专家 demo**。带走:
- **世界模型学转移** (从随机数据!), 学会后能想象 + 规划 (MPC) → 不学策略也能控制。
- **数据优势**: 随机探索数据就能学世界模型 (vs BC 需专家) — model-based 的杀手锏。
- **误差累积** (M13.5): 想象长了不准 → MPC 用短视野 + 每步重规划纠偏。
下一步 N2: 量化 model-based vs model-free 的样本效率。"""),
]
nbformat.write(n1, HERE / "N1-world-model-planning.ipynb")
print("written N1")

# ── N2 model-based-vs-free ──
n2 = new_notebook()
n2.cells = [
    md("""# N2 · model-based vs model-free 样本效率

> 配套 11.5-L3 · 对比两条路:
> - **model-based**: 从**随机**转移学世界模型 + MPC 规划 (数据便宜, 零专家)
> - **model-free (BC)**: 从**专家** demo 监督学策略 (需专家)
> 看 model-based 怎么用便宜数据换好策略。"""),
    code(PATHS + """
# 复用 M11.4 的 BC (model-free 对照)
sys.path.insert(0, str(Path.cwd().parents[1] / "robot-data-imitation" / "src"))
import bc_train as bc
print('就绪: world_model (model-based) + bc_train (model-free)')"""),
    md("## 1. 两条路在不同数据量下的成功率"),
    code("""torch.manual_seed(0)
sizes = [20, 50, 100, 500, 2000]
mb, mf = [], []
for n in sizes:
    # model-based: n 条随机转移 → 世界模型 → MPC
    S,A,D = wm.make_random_transitions(n=n, seed=0)
    m = wm.build_world_model(); wm.train_world_model(m, S, A, D, epochs=400)
    mb.append(env.eval_policy(wm.mpc_policy_fn(m, n_samples=150, horizon=6), n_episodes=80))
    # model-free: 用约 n 个专家 (state,action) 对做 BC
    Se,Ae = env.make_demos(n=max(2, n//8), seed=0)   # 每条 demo ~8 步, 凑到 ~n 个对
    Se,Ae = Se[:n], Ae[:n]
    mfm = bc.build_bc_policy(); bc.train_bc(mfm, Se, Ae, epochs=300)
    mf.append(env.eval_policy(bc.bc_policy_fn(mfm), n_episodes=80))
    print(f'{n:5d} 样本: model-based(随机) {mb[-1]:.2f} | model-free/BC(专家) {mf[-1]:.2f}')"""),
    md("## 2. 样本效率曲线"),
    code(MPL + """
fig, ax = plt.subplots(figsize=(7.5,4.3))
ax.plot(sizes, mb, 'o-', color='C0', label='model-based (世界模型+MPC, 随机数据)')
ax.plot(sizes, mf, 's--', color='C3', label='model-free (BC, 专家数据)')
ax.set_xscale('log'); ax.set_xlabel('样本数 (log)'); ax.set_ylabel('成功率'); ax.set_ylim(0,1.05)
ax.legend(); ax.set_title('样本效率: model-based 用随机(便宜)数据也能解任务')
plt.tight_layout(); plt.show()
print('model-based(随机):', {n:round(v,2) for n,v in zip(sizes,mb)})
print('model-free(专家): ', {n:round(v,2) for n,v in zip(sizes,mf)})
print('→ 关键: model-based 的数据是随机探索(免费); model-free 的数据是专家(贵)。')"""),
    md("## 3. 解读 (接 L3 + 9.4)"),
    code(r"""lines = [
 'model-based vs model-free (L3):',
 '  model-based: 学世界模型(随机数据可) + 规划 -> 数据便宜, 样本效率高',
 '               但 世界模型不准->规划歪 (model bias); 复杂动态难学',
 '  model-free:  直接学策略 -> 简单通用; 但 BC 需专家 / RL 需大量交互',
 '',
 '关键差异不只是成功率曲线, 更是数据从哪来:',
 '  - model-based 从随机探索学 (本 toy reach 动态好学 -> MPC 高成功)',
 '  - model-free 从专家 demo 学 (数据贵)',
 '  -> 机器人(交互贵)偏好 model-based 用便宜数据换策略; 复杂任务世界模型难学, 常混合。',
]
print(chr(10).join(lines))"""),
    md("""## 4. 反思 (11.5 收口)

你对比了两条路的样本效率。带走:
- **model-based**: 世界模型(随机数据)+规划, 数据便宜、样本效率高; 但靠世界模型准 (复杂动态难学)。
- **model-free (BC)**: 直接学策略, 简单; 但需专家 demo (贵)。
- 核心差异是**数据从哪来**: 随机探索(免费) vs 专家(贵); 机器人(交互贵)偏好 model-based, 常混合。

> **M11.5 收口**: 世界模型学转移→想象→规划(MPC); 先学想象再学行动; model-based 省交互; 视频模型=图像世界模型 (接 M13)。
> **交棒 M11.6「sim2real-isaaclab」**: 从想象的世界到**仿真**的世界 —— IsaacLab 仿真训练 + sim2real gap (你碰过 IsaacLab!)。下一专题 `sim2real-isaaclab`。"""),
]
nbformat.write(n2, HERE / "N2-model-based-vs-free.ipynb")
print("written N2")
