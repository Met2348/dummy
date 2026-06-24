"""生成 11.3 notebooks (N1 扩散动作头多峰 / N2 action chunking 消融). 跑后 nbconvert --execute。"""
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
import diffusion_policy as dp
import numpy as np, torch"""

def obstacle_patch():
    return ("circ = plt.Circle((dp.OBST_C[0], dp.OBST_C[1]), dp.OBST_R, color='gray', alpha=0.4)\n"
            "ax.add_patch(circ)")

# ── N1 diffusion-action-head ──
n1 = new_notebook()
n1.cells = [
    md("""# N1 · 扩散动作头解决多峰 (vs 回归取平均)

> 配套 11.3-L1/L2 · 玩具: 绕障碍导航, 上绕/下绕都行 (双峰)。
> 训扩散动作头 + 连续回归头, 在障碍前采样动作:
> **扩散是双峰 (上/下), 回归塌成中间一点 (直冲障碍)**。这是 diffusion policy 的核心价值。"""),
    code(PATHS + "\nprint('环境: 绕障碍导航, 障碍中心', dp.OBST_C, '半径', dp.OBST_R)"),
    md("## 1. 双峰专家 demo: 一半上绕、一半下绕"),
    code(MPL + """
import numpy as np
# 画几条专家轨迹 (上绕/下绕)
fig, ax = plt.subplots(figsize=(5.5,5))
for i in range(6):
    side = 1.0 if i%2==0 else -1.0
    pos = np.array([dp.START_X, 0.0], np.float32); tr=[pos.copy()]
    for _ in range(dp.MAX_T):
        tgt = dp._expert_target(pos, side); d=tgt-pos; n=np.linalg.norm(d)
        a = d/n if n>1e-6 else np.zeros(2); pos = pos + a*dp.STEP; tr.append(pos.copy())
        if dp.reached(pos): break
    tr=np.array(tr); ax.plot(tr[:,0],tr[:,1],'-',alpha=0.7,color='C0' if side>0 else 'C1')
""" + obstacle_patch() + """
ax.plot(dp.START_X,0,'gs',ms=12,label='起点'); ax.plot(*dp.GOAL,'r*',ms=18,label='目标')
ax.legend(); ax.set_title('双峰专家: 上绕(蓝)/下绕(橙) 都合理'); ax.set_aspect('equal')
plt.tight_layout(); plt.show()
print('→ 同一起点, 两个同样合理的动作 (上/下绕) = 多峰动作分布。')"""),
    md("## 2. 训扩散动作头 + 连续回归头"),
    code("""torch.manual_seed(0)
S, A = dp.make_obstacle_demos(n=400, chunk=1, seed=0)
print(f'双峰专家 demo: {len(S)} 个 (state, action) 对')
diff = dp.build_diffusion_policy(chunk=1)
_, sched = dp.train_diffusion_policy(diff, S, A, epochs=800)
reg = dp.build_regression(chunk=1); dp.train_regression(reg, S, A, epochs=800)
print('扩散动作头 + 回归头 训练完毕')"""),
    md("## 3. 关键对比: 障碍前的动作分布 (扩散双峰 vs 回归取平均)"),
    code(MPL + """
st = np.array([dp.START_X, 0.0], np.float32)
acts = np.array([dp.sample_action(diff, st, sched, seed=k)[0] for k in range(300)])
reg_a = reg(torch.tensor(st[None])).detach().numpy()[0]
fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(acts[:,0], acts[:,1], s=14, alpha=0.4, color='C0', label='扩散采样 (300)')
ax.scatter([reg_a[0]],[reg_a[1]], s=240, marker='X', color='red', label='回归 (单点=均值)', zorder=5)
ax.axhline(0, ls=':', c='gray')
ax.set_xlabel('动作 dx'); ax.set_ylabel('动作 dy')
ax.set_title('障碍前的动作分布: 扩散双峰(上/下) vs 回归塌成均值(y≈0)'); ax.legend()
plt.tight_layout(); plt.show()
up=(acts[:,1]>0.15).sum(); dn=(acts[:,1]<-0.15).sum()
print(f'扩散采样: 朝上 {up}, 朝下 {dn} (双峰!); 回归动作 dy={reg_a[1]:+.2f} (≈0=取平均, 正对障碍)')
print('→ 回归取均值 = 既不上也不下 = 直冲障碍 (多峰死穴); 扩散采样到某一峰 → 能绕开。')"""),
    md("## 4. rollout: 扩散绕开 (采到一侧), 回归易直冲"),
    code(MPL + """
fig, axes = plt.subplots(1,2,figsize=(11,5))
for ax, (name, fn) in zip(axes, [('扩散动作头', dp.make_diffusion_action_fn(diff, sched)),
                                  ('连续回归', dp.make_regression_action_fn(reg))]):
    for seed in range(6):
        ok,c,steps,tr = dp.rollout(fn, seed=seed, record=True); tr=np.array(tr)
        ax.plot(tr[:,0],tr[:,1],'-',alpha=0.7,color=('C2' if ok else 'C3'))
    circ = plt.Circle((dp.OBST_C[0],dp.OBST_C[1]), dp.OBST_R, color='gray', alpha=0.4); ax.add_patch(circ)
    ax.plot(dp.START_X,0,'gs',ms=10); ax.plot(*dp.GOAL,'r*',ms=16)
    sr,cr = dp.eval_policy(fn, n_episodes=100)
    ax.set_title(f'{name}\\n成功 {sr:.2f}, 撞障 {cr:.2f}'); ax.set_aspect('equal')
plt.suptitle('绿=成功, 红=撞障; 扩散采一侧绕开, 回归动作正对障碍'); plt.tight_layout(); plt.show()
print('→ 扩散动作头表达多峰=能绕; 回归取平均=直对障碍。这就是 diffusion policy 的价值。')"""),
    md("""## 5. 反思
你看到扩散动作头解决了多峰矛盾。带走:
- **回归取均值**: 多峰的均值落谷里 (直冲障碍) — 数学必然 (L1)。
- **扩散建模分布 + 采样**: 采到某一峰 (上/下绕) → 连续 + 多峰两全。
- 机制 = M13.1 DDPM + state 条件 (同源, L2); 你已会全部。
> 注意: 单步扩散每步独立采样, 可能来回切模式 (抖动) → 这是 N2 action chunking 要治的。
下一步 N2: action chunking 的平滑/反应性权衡。"""),
]
nbformat.write(n1, HERE / "N1-diffusion-action-head.ipynb")
print("written N1")

# ── N2 action-chunking ──
n2 = new_notebook()
n2.cells = [
    md("""# N2 · action chunking 消融: 平滑 vs 反应性

> 配套 11.3-L3 · action chunking = 一次预测 H 步动作 (开环执行再重规划)。
> 消融 chunk 大小, 量「没有免费的 chunk」: chunk ↑ → 块内更连贯, 但开环执行失去闭环纠错 →
> 反应性↓ (成功率可能掉)。这是 chunking 的核心权衡 (接 9.4 实验设计)。"""),
    code(PATHS + "\nprint('就绪')"),
    md("## 1. 训练不同 chunk 大小的扩散动作头"),
    code("""torch.manual_seed(0)
CHUNKS = [1, 2, 4, 8]
models = {}
for ch in CHUNKS:
    S, A = dp.make_obstacle_demos(n=300, chunk=ch, seed=0)
    m = dp.build_diffusion_policy(chunk=ch)
    _, sc = dp.train_diffusion_policy(m, S, A, epochs=600)
    models[ch] = (m, sc)
    print(f'chunk={ch} 训练完毕 (动作块维度 {ch*dp.ACT_DIM})')"""),
    md("## 2. 量化: chunk vs 成功率(反应性) + 执行轨迹平滑度"),
    code("""def traj_smoothness(fn, seeds=range(40)):
    # 执行轨迹平滑度 = 相邻位移方向变化的平均 (越小越平滑/连贯)
    changes=[]
    for sd in seeds:
        _,_,_,tr = dp.rollout(fn, seed=int(sd), record=True)
        tr=np.array(tr)
        if len(tr)<3: continue
        steps=np.diff(tr,axis=0);
        for i in range(1,len(steps)):
            a,b=steps[i-1],steps[i]
            na,nb=np.linalg.norm(a),np.linalg.norm(b)
            if na>1e-6 and nb>1e-6:
                changes.append(1-np.clip((a@b)/(na*nb),-1,1))
    return float(np.mean(changes)) if changes else 0.0

rows=[]
for ch in CHUNKS:
    m, sc = models[ch]
    fn = dp.make_diffusion_action_fn(m, sc)
    sr, cr = dp.eval_policy(fn, n_episodes=100)
    sm = traj_smoothness(fn)
    rows.append((ch, sr, cr, sm))
    print(f'chunk={ch}: 成功率 {sr:.2f}, 撞障率 {cr:.2f}, 轨迹不平滑度 {sm:.3f}')"""),
    md("## 3. 可视化权衡曲线"),
    code(MPL + """
chs=[r[0] for r in rows]; srs=[r[1] for r in rows]; sms=[r[3] for r in rows]
fig, ax1 = plt.subplots(figsize=(7,4.2))
ax1.plot(chs, srs, 'o-', color='C2', label='成功率 (反应性)')
ax1.set_xlabel('action chunk 大小 H'); ax1.set_ylabel('成功率', color='C2'); ax1.set_ylim(0,1.05)
ax2 = ax1.twinx(); ax2.plot(chs, sms, 's--', color='C3', label='轨迹不平滑度')
ax2.set_ylabel('不平滑度 (越低越连贯)', color='C3')
plt.title('action chunking 权衡: chunk↑ 改变连贯性, 但开环执行→反应性↓')
fig.tight_layout(); plt.show()
print('measured:', {r[0]:f'成功{r[1]:.2f}/不平滑{r[3]:.2f}' for r in rows})
print('→ 本玩具需要闭环纠错, chunk 越大开环越久→成功率下降 (反应性代价)。')"""),
    md("""## 4. 反思 (11.3 收口)

你量化了 action chunking 的核心权衡。带走:
- **chunking 不是免费的**: 一次预测一段→块内连贯/抗抖/抗延迟, 但**开环执行一长段→失去闭环纠错** (反应性↓)。
- 本玩具需随时纠偏, 所以大 chunk 成功率掉; 真实任务要按「多频繁纠错」折中 (常配 temporal ensembling)。
- chunk 大小是个**任务相关的旋钮** (同扩散步数/CFG, 没有万能值, 该消融, 接 9.4)。

> **M11.3 收口**: 扩散动作头解决多峰(N1); flow-matching 解决快(L3); chunking 解决稳但权衡反应性(N2)。π 动作头 = M10 VLM + M13 flow/扩散 + chunking。
> **交棒 M11.4「robot-data-imitation」**: 动作头怎么训? **模仿学习** (从专家 demo) + 机器人数据 scaling 教训。下一专题 `robot-data-imitation`。"""),
]
nbformat.write(n2, HERE / "N2-action-chunking.ipynb")
print("written N2")
