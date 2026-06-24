"""生成 11.4 notebooks (N1 BC + 分布漂移 / N2 数据 scaling 曲线). 跑后 nbconvert --execute。"""
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
import bc_train as bc
import toy_env as env   # M11.1 共享 (bc_train 已加进 path)
import numpy as np, torch"""

# ── N1 behavior-cloning ──
n1 = new_notebook()
n1.cells = [
    md("""# N1 · 行为克隆 (BC) + 分布漂移

> 配套 11.4-L1 · BC = 从专家 demo 监督学策略 (state→action 回归)。
> 训 BC 让它走到目标; 再看 **demo 太少时的分布漂移** (走到专家没覆盖的状态就乱)。"""),
    code(PATHS + "\nprint('环境 = M11.1 2D 到达任务; BC = state→action 监督回归')"),
    md("## 1. 行为克隆: 从专家 demo 监督学策略"),
    code("""torch.manual_seed(0)
S, A = env.make_demos(n=200, seed=0)
print(f'专家 demo: {len(S)} 个 (state, action) 对')
model = bc.build_bc_policy()
losses = bc.train_bc(model, S, A, epochs=300)
sr = env.eval_policy(bc.bc_policy_fn(model), n_episodes=200)
print(f'BC 训练 loss {losses[0]:.3f} → {losses[-1]:.3f}; 成功率 {sr:.2f}')"""),
    md("## 2. BC 策略 rollout (学会了模仿专家朝目标走)"),
    code(MPL + """
pol = bc.bc_policy_fn(model)
fig, axes = plt.subplots(1,3,figsize=(13,4.2))
for ax, seed in zip(axes, [4,8,12]):
    ok,steps,traj = env.rollout(pol, seed=seed, record=True); traj=np.array(traj)
    ax.plot(traj[:,0],traj[:,1],'-o',ms=4); ax.plot(traj[0,0],traj[0,1],'gs',ms=12); ax.plot(traj[0,2],traj[0,3],'r*',ms=16)
    ax.add_patch(plt.Rectangle((-1,-1),2,2,fill=False,ls=':',ec='gray'))
    ax.set_title(f'BC (成功={ok}, {steps}步)'); ax.set_aspect('equal')
plt.suptitle('行为克隆: 监督学专家 → 学会朝目标走'); plt.tight_layout(); plt.show()"""),
    md("## 3. 分布漂移: demo 太少 vs 充足"),
    code(MPL + """
fig, axes = plt.subplots(1,2,figsize=(11,4.5))
for ax, ndemo in zip(axes, [3, 200]):
    Sd, Ad = env.make_demos(n=ndemo, seed=0)
    m = bc.build_bc_policy(seed=0); bc.train_bc(m, Sd, Ad, epochs=300, seed=0)
    p = bc.bc_policy_fn(m); sr = env.eval_policy(p, n_episodes=200)
    for seed in range(6):
        ok,steps,traj = env.rollout(p, seed=seed, record=True); traj=np.array(traj)
        ax.plot(traj[:,0],traj[:,1],'-',alpha=0.6,color='C2' if ok else 'C3')
        ax.plot(traj[0,2],traj[0,3],'r*',ms=12)
    ax.add_patch(plt.Rectangle((-1,-1),2,2,fill=False,ls=':',ec='gray'))
    ax.set_title(f'{ndemo} 条 demo (成功率 {sr:.2f})'); ax.set_aspect('equal')
plt.suptitle('分布漂移: demo 少→覆盖不到的状态乱走(红); demo 多→稳(绿)'); plt.tight_layout(); plt.show()
print('→ demo 太少, BC 在专家没覆盖的状态没学过 → 漂移/失败 (L1 复合误差)。解法: 更多更广数据 (L3)。')"""),
    md("""## 4. 反思
你做了行为克隆并看到分布漂移。带走:
- **BC = 监督学专家** (state→action), 机器人训练的基础 (你最熟的范式)。
- **分布漂移**: 偏出专家覆盖的状态就乱 (复合误差, 同 M13.5 误差累积)。
- 解法主力 = **数据 scaling** (更多更广 demo, N2 量化)。
下一步 N2: 数据量 vs 成功率曲线 (scaling)。"""),
]
nbformat.write(n1, HERE / "N1-behavior-cloning.ipynb")
print("written N1")

# ── N2 data-scaling ──
n2 = new_notebook()
n2.cells = [
    md("""# N2 · 数据 scaling 曲线: demo 数量 vs 成功率

> 配套 11.4-L3 · 把 LLM 的 scaling 教训搬到机器人: 更多 demo → 更高成功率。
> 画 scaling 曲线, 找「够用」的数据量拐点。固定其它变量 (模型/训练步/评估集) 做干净对比 (接 9.4)。"""),
    code(PATHS + "\nprint('就绪')"),
    md("## 1. 不同 demo 数量训 BC, 测成功率 (固定其它变量)"),
    code("""torch.manual_seed(0)
sizes, srs, npairs = bc.scaling_curve(demo_sizes=(2,5,10,25,50,100,200), epochs=300, seed=0)
print(f"{'demo数':>6} {'(s,a)对':>8} {'成功率':>8}")
for n, sr, npr in zip(sizes, srs, npairs):
    print(f"{n:6d} {npr:8d} {sr:8.2f}")"""),
    md("## 2. scaling 曲线 (上升渐饱和, 同 LLM)"),
    code(MPL + """
fig, ax = plt.subplots(figsize=(7,4.3))
ax.plot(sizes, srs, 'o-', color='C0')
ax.set_xscale('log'); ax.set_xlabel('专家 demo 数量 (log)'); ax.set_ylabel('成功率'); ax.set_ylim(0,1.05)
ax.axhline(0.9, ls='--', c='gray', alpha=0.6)
# 标注"够用"拐点 (首个 ≥0.9)
import numpy as np
knee = next((n for n,s in zip(sizes,srs) if s>=0.9), None)
if knee: ax.axvline(knee, ls=':', c='C2'); ax.text(knee, 0.3, f'够用拐点\\n≈{knee} demo', color='C2')
ax.set_title('机器人数据 scaling: 更多 demo → 更高成功率 (渐饱和)')
plt.tight_layout(); plt.show()
print(f'→ 少 demo 成功率低 (分布漂移); ~{knee} 条后接近饱和。数据是机器人最大杠杆 (L3)。')"""),
    md("## 3. 解读 (接 9.4 实验设计)"),
    code("""print('''scaling 实验的干净做法 (接 9.4):
  - 固定: 模型大小、训练步数、评估集、种子 → 只变 demo 数量 (单变量)
  - 否则: 混入其它变量, 曲线不可信 (因果不干净)

读这条曲线 (接 LLM scaling 直觉):
  - 上升段: 数据是瓶颈, 加数据立竿见影
  - 饱和段: 数据够了, 瓶颈转移 (模型容量/任务难度/数据多样性)
  - "够用拐点": 工程上性价比最高的数据量 (再加收益递减)

机器人现状: 多数任务还在"上升段" (数据远不够) → 数据是最大机会 (L3)。''')"""),
    md("""## 4. 反思 (11.4 收口)

你画了机器人的数据 scaling 曲线。带走:
- **数据量 → 成功率**: 上升渐饱和, 同 LLM scaling; 少数据→分布漂移→低成功率。
- **干净 scaling 实验** (接 9.4): 单变量 (只变数据量), 固定其它。
- **多样性 > 单纯量** (L3): toy 的随机起点/目标 = 天然多样, 故少量就泛化。
- 机器人现状多在「上升段」: 数据是最大瓶颈也是最大机会。

> **M11.4 收口**: BC 是训练基础 (分布漂移); 数据靠遥操作 (贵) + scaling + co-train + 人类视频。
> **交棒 M11.5「world-action-models」**: 从纯模仿到带想象 —— 具身世界模型, 接你 M13.5 的 `world_model.py`。下一专题 `world-action-models`。"""),
]
nbformat.write(n2, HERE / "N2-data-scaling.ipynb")
print("written N2")
