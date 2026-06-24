"""生成 11.1 notebooks (N1 环境+序列化 / N2 具身版 next-action 预测). 跑后 nbconvert --execute。"""
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
import toy_env as env, action_serialize as ser
import numpy as np"""

# ── N1 env-and-serialize ──
n1 = new_notebook()
n1.cells = [
    md("""# N1 · 玩具控制环境 + tokens-as-actions 序列化

> 配套 11.1-L1/L3 · 跑一个 2D 到达任务的闭环 (状态→动作→转移), 再把动作离散成 token。
> 看具身的「感知-行动闭环」和 RT-2 的「动作当 token」最小形态。"""),
    code(PATHS + "\nprint('环境就绪: state', env.STATE_DIM, 'D, action', env.ACT_DIM, 'D; 动作 token', ser.N_ACTION_TOKENS, '个')"),
    md("## 1. 闭环: 专家策略走到目标 (状态→动作→转移)"),
    code(MPL + """
ok, steps, traj = env.rollout(env.expert_action, seed=2, record=True)
traj = np.array(traj)
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(traj[:,0], traj[:,1], '-o', ms=5, label='agent 轨迹')
ax.plot(traj[0,0], traj[0,1], 'gs', ms=14, label='起点')
ax.plot(traj[0,2], traj[0,3], 'r*', ms=20, label='目标')
ax.add_patch(plt.Rectangle((-1,-1),2,2,fill=False,ls=':',ec='gray'))
ax.legend(); ax.set_title(f'2D 到达任务: 专家策略 (成功={ok}, {steps}步)'); ax.set_aspect('equal')
plt.tight_layout(); plt.show()
print(f'专家策略成功率 (200 episodes): {env.eval_policy(env.expert_action, n_episodes=200):.2f}')"""),
    md("## 2. tokens-as-actions: 把连续动作离散成 token"),
    code(MPL + """
S, A = env.make_demos(n=200, seed=0)
states, toks = ser.serialize_episode(S, A)
print(f'序列化: {len(S)} 个 (state, action) → 动作 token 序列')
print(f'离散化损失 (方向余弦差): {ser.roundtrip_error(A):.3f} (8 方向量化的固有代价)')
# 动作 token 分布
fig, ax = plt.subplots(figsize=(7,3.6))
counts = np.bincount(toks, minlength=ser.N_ACTION_TOKENS)
names = ['→','↗','↑','↖','←','↙','↓','↘','停']
ax.bar(names, counts, color='C0')
ax.set_title('专家动作的离散 token 分布 (8方向+停)'); ax.set_ylabel('次数')
plt.tight_layout(); plt.show()
print('→ 动作变成了 token (像词)。于是"控制"就能当"预测下一个 token"做 (RT-2 范式, N2 验证)。')"""),
    md("""## 3. 反思
你跑了具身的**感知-行动闭环**, 并把动作序列化成 token。带走:
- 闭环: 状态 → 策略 π → 动作 → 环境转移 → 新状态 (具身比 LLM/VLM 多的部分)。
- **tokens-as-actions**: 连续动作离散成 token, 控制就变成 token 预测 (RT-2 内核)。
- 离散化有损 (方向被量化), 是这个范式的固有代价 (推动 M11.3 扩散动作头)。
下一步 N2: 训一个 tiny transformer 做「状态→下一动作 token」预测 (具身版 next-token)。"""),
]
nbformat.write(n1, HERE / "N1-env-and-serialize.ipynb")
print("written N1")

# ── N2 next-action-prediction ──
n2 = new_notebook()
n2.cells = [
    md("""# N2 · 具身版 next-token: 状态 → 下一动作 token

> 配套 11.1-L3 · 训一个 tiny 模型, 给状态预测下一个动作 token (9 类分类 = next-token)。
> 然后用它当策略 rollout, 看学到的策略能不能到达目标。这是 RT-2「控制=预测token」的玩具内核。"""),
    code(PATHS + "\nimport torch, torch.nn as nn\nprint('就绪')"),
    md("## 1. 从专家 demo 学一个「状态→动作 token」预测器 (模仿学习)"),
    code("""torch.manual_seed(0)
S, A = env.make_demos(n=400, seed=0)
toks = np.array([ser.action_to_token(a) for a in A], dtype=np.int64)
X = torch.tensor(S); Y = torch.tensor(toks)
# tiny 策略网络: state(4) → 9 个动作 token 的 logits (具身版 next-token 预测头)
policy = nn.Sequential(nn.Linear(env.STATE_DIM, 64), nn.SiLU(),
                       nn.Linear(64, 64), nn.SiLU(),
                       nn.Linear(64, ser.N_ACTION_TOKENS))
opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
for ep in range(400):
    logits = policy(X)
    loss = nn.functional.cross_entropy(logits, Y)
    opt.zero_grad(); loss.backward(); opt.step()
acc = (policy(X).argmax(-1) == Y).float().mean().item()
print(f'训练 loss {loss.item():.3f}, 训练集动作 token 准确率 {acc:.2f}')"""),
    md("## 2. 用学到的策略 rollout: 它能走到目标吗?"),
    code(MPL + """
def learned_policy(state):
    with torch.no_grad():
        tok = policy(torch.tensor(state[None], dtype=torch.float32)).argmax(-1).item()
    return ser.token_to_action(tok)

sr = env.eval_policy(learned_policy, n_episodes=200)
sr_expert = env.eval_policy(env.expert_action, n_episodes=200)
print(f'学到的策略成功率: {sr:.2f}  (专家上界 {sr_expert:.2f})')
# 画几条学到策略的轨迹
fig, axes = plt.subplots(1, 3, figsize=(13,4.2))
for ax, seed in zip(axes, [5, 7, 11]):
    ok, steps, traj = env.rollout(learned_policy, seed=seed, record=True)
    traj = np.array(traj)
    ax.plot(traj[:,0], traj[:,1], '-o', ms=4)
    ax.plot(traj[0,0], traj[0,1], 'gs', ms=12); ax.plot(traj[0,2], traj[0,3], 'r*', ms=16)
    ax.add_patch(plt.Rectangle((-1,-1),2,2,fill=False,ls=':',ec='gray'))
    ax.set_title(f'学到策略 (成功={ok}, {steps}步)'); ax.set_aspect('equal')
plt.suptitle('具版 next-token 策略: 学会了朝目标走'); plt.tight_layout(); plt.show()
print('→ 一个 next-action 预测器 = 一个能用的策略。这就是 RT-2 的玩具内核 (控制=预测token)。')"""),
    md("""## 3. 反思 (11.1 收口)

你训了一个「状态→下一动作 token」预测器, 它就是一个能到达目标的策略。带走:
- **控制 = next-token 预测**: 和你 LLM 一模一样的机制 (CE 训练 + argmax 解码), 只是 token 是动作。
- **模仿学习**: 从专家 demo 学策略 = 监督学习 (M11.4 会深入)。
- 离散动作 token 够走到目标, 但精度受限于离散化 (推动 M11.3 连续/扩散动作头)。

> **M11.1 收口**: 具身闭环 + tokens-as-actions + 数据正迁移; 控制能当 next-token 做 = 复用你 LLM 全套。
> **交棒 M11.2「vla-architectures」**: 把感知核换成真正的 VLM —— 用 M10 的 mini-VLM + 动作头组装 mini-VLA。下一专题 `vla-architectures`。"""),
]
nbformat.write(n2, HERE / "N2-next-action-prediction.ipynb")
print("written N2")
