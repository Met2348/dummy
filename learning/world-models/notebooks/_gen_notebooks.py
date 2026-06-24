"""生成 13.5 notebooks (N1 学世界模型+想象 rollout / N2 多步误差累积). 跑后 nbconvert --execute。"""
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

# ── N1 learn-world-model ──
n1 = new_notebook()
n1.cells = [
    md("""# N1 · 学一个世界模型 + 脑内想象 rollout

> 配套 13.5-L1 · 玩具环境: 2D 智能体在方盒里, 动作=上下左右各推一步。
> 训一个世界模型学 (状态,动作)→Δ状态, 然后给它一串动作让它**想象**整条轨迹, 和真环境对照。
> 亲眼看「模型学会了世界怎么动」。"""),
    code(PATHS + "\nimport numpy as np, torch\nimport world_model as wm\nprint('world_model 就绪, 动作数', wm.N_ACTIONS)"),
    md("## 1. 采转移样本 (state, action → Δstate) 训世界模型"),
    code("""torch.manual_seed(0)
data = wm.make_transitions(n=4000, seed=0)
print('转移样本:', data[0].shape, '个 (状态, 动作, Δ状态)')
model = wm.build_world_model()
losses = wm.train_world_model(model, data, epochs=400)
print(f'世界模型训练: loss {losses[0]:.4f} → {losses[-1]:.4f}')"""),
    md("## 2. 脑内想象 vs 真环境: 同一串动作, 看想象轨迹贴不贴真实"),
    code(MPL + """
start = np.array([0.0, 0.0], np.float32)
acts = [3,3,3,0,0,2,2,0,3,0]      # 一串动作 (右右右上上左左上右上)
img = wm.imagine(model, start, acts)        # 脑内想象 (不碰真环境)
tru = wm.true_rollout(start, acts, seed=1)  # 真环境
fig, ax = plt.subplots(figsize=(5.5,5))
ax.plot(tru[:,0], tru[:,1], '-o', label='真环境 rollout', color='gray', ms=6)
ax.plot(img[:,0], img[:,1], '--s', label='世界模型想象', color='C0', ms=5)
ax.plot(*start, 'g*', ms=18, label='起点')
ax.add_patch(plt.Rectangle((-1,-1),2,2,fill=False,ls=':',ec='r'))
ax.legend(); ax.set_title('世界模型「脑内想象」紧贴真环境'); ax.set_aspect('equal')
plt.tight_layout(); plt.show()
print(f'想象终点 {img[-1].round(3)}  真实终点 {tru[-1].round(3)}  末端差 {np.linalg.norm(img[-1]-tru[-1]):.3f}')
print('→ 世界模型学会了环境动态: 给动作就能想象未来, 不需要碰真环境 (model-based 规划的地基)。')"""),
    md("""## 3. 反思
你训了一个世界模型并用它**想象**了未来。带走:
- 世界模型 = 学转移 (状态,动作)→下一状态; 学会后能脑内 rollout, 不碰真环境。
- 实现智慧: 预测 **Δ状态 (变化量)** 比预测绝对状态更稳 (同扩散学噪声/ResNet 残差)。
- 用途: model-based 规划 (在想象里试动作)、省真实交互、机器人预演 (接 M11)。
下一步 N2: 想象越长越不准吗? 量化**多步误差累积** (世界模型核心难题)。"""),
]
nbformat.write(n1, HERE / "N1-learn-world-model.ipynb")
print("written N1")

# ── N2 prediction-quality ──
n2 = new_notebook()
n2.cells = [
    md("""# N2 · 世界模型预测质量: 多步误差累积

> 配套 13.5-L1/L4 · 想象 rollout 是自回归的 —— 每步小误差会喂进下一步, 复利式放大。
> 量化「想象 vs 真实」的误差随步数怎么增长。这是世界模型 (和长程视频/自回归预测) 的核心难题。"""),
    code(PATHS + "\nimport numpy as np, torch\nimport world_model as wm\nprint('就绪')"),
    md("## 1. 训世界模型, 测多步误差增长曲线"),
    code("""torch.manual_seed(0)
data = wm.make_transitions(n=4000, seed=0)
model = wm.build_world_model()
wm.train_world_model(model, data, epochs=400)
errs = wm.multistep_error(model, n_traj=80, horizon=25)
for h in [1,5,10,15,20,25]:
    print(f'  {h:2d} 步想象: 平均误差 {errs[h]:.3f}')
print('→ 1 步很准, 步数越多误差越大 (复利累积)。')"""),
    md("## 2. 误差累积曲线 (短期可信, 长期发散)"),
    code(MPL + """
fig, ax = plt.subplots(figsize=(7,4.2))
H = len(errs)-1
ax.plot(range(H+1), errs, '-o', ms=4, color='C3')
ax.fill_between(range(H+1), 0, errs, alpha=0.15, color='C3')
ax.set_xlabel('想象步数 (rollout horizon)'); ax.set_ylabel('想象 vs 真实 平均误差')
ax.set_title('世界模型多步误差累积: 短期可信, 长期发散')
ax.axvspan(0,5,alpha=0.1,color='green'); ax.text(2.5, errs.max()*0.9, '短期\\n可信', ha='center', color='green')
plt.tight_layout(); plt.show()
print('→ 这是 model-based 规划用「短视野 + 重规划 (MPC) + 真观测纠偏」对抗的根本难题 (L1/L4)。')"""),
    md("## 3. 1 步预测其实很准 (问题在累积, 不在单步)"),
    code("""# 单步预测误差 (不累积): 直接在真转移上测
rng = np.random.default_rng(7)
s = rng.uniform(-1,1,size=(500,2)).astype(np.float32)
a = rng.integers(0, wm.N_ACTIONS, size=500)
true_next = np.stack([wm.true_step(s[i], int(a[i]), rng) for i in range(500)])
import torch
with torch.no_grad():
    onehot = torch.nn.functional.one_hot(torch.tensor(a), wm.N_ACTIONS).float()
    pred_delta = model(torch.tensor(s), onehot).numpy()
pred_next = np.clip(s + pred_delta, -1, 1)
single = np.linalg.norm(pred_next - true_next, axis=-1).mean()
print(f'单步预测误差: {single:.4f} (很小)')
print(f'25 步累积误差: {errs[25]:.4f} (是单步的 {errs[25]/single:.0f}×)')
print('→ 世界模型单步很准; 难点是自回归把小误差滚成大偏差。这正是长程一致 (M13.4-L3) 的同一根难题。')"""),
    md("""## 4. 反思 (13.5 收口)

你量化了世界模型的核心难题: **多步误差累积**。带走:
- **单步准 ≠ 多步准**: 自回归想象把每步小误差复利放大, 长程发散。
- 这和 M13.4-L3 的长程视频不一致是**同一根难题** (自回归预测的通病)。
- 对抗手段: 短规划视野、MPC 重规划、周期性用真观测纠偏 (接 RL/M11)。
- 评测世界模型 = 看误差增长曲线 (但「懂物理 vs 拟合」更深的评测仍是 open, L4)。

> **M13.5 收口**: 世界模型 = 动作条件的生成式预测; 能想象 rollout (规划地基); 误差累积是主敌; 它是生成(M13)与具身(M11)的共享内核。
> **交棒 M13.6「diffusion-language-models」**: 生成式媒体绕一圈回你本行 —— 文本也能扩散 (masked diffusion LM, 并行生成)。下一专题 `diffusion-language-models`。"""),
]
nbformat.write(n2, HERE / "N2-prediction-quality.ipynb")
print("written N2")
