"""为 6 个 nb=0 RL/后训练旧专题回补 notebook (复用既有 mock-data src)。
运行: python learning/_shared/gen_rl_notebooks.py"""
from __future__ import annotations
import sys
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LEARNING = Path(__file__).resolve().parent.parent
def md(s): return new_markdown_cell(s)
def code(s): return new_code_cell(s)
MPL = """import matplotlib, matplotlib.pyplot as plt
matplotlib.rcParams['axes.unicode_minus']=False
for f in ['Microsoft YaHei','SimHei','DejaVu Sans']:
    try: matplotlib.rcParams['font.sans-serif']=[f]; break
    except Exception: pass"""
PRE = """import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
import numpy as np, torch"""

def write(topic, fname, cells):
    d = LEARNING / topic / "notebooks"; d.mkdir(parents=True, exist_ok=True)
    nb = new_notebook(); nb.cells = cells
    nbformat.write(nb, d / fname); print(f"written {topic}/notebooks/{fname}")


# 1. dpo-family: 6 个偏好优化变体横向对比
write("dpo-family", "N1-po-variants-comparison.ipynb", [
    md("""# N1 · 6 个偏好优化变体横向对比 (DPO/IPO/ORPO/SimPO/CPO/DPOP)

> 复用 `src/capstone_dpo_comparison.py` · 统一框架下对比 6 个 DPO 家族变体的训练动态 (margin 变化)。
> (mock 动态, 教学用; 真实训练用 gpt2, 见各 `*_minimal.py`)。"""),
    code(PRE + "\nimport capstone_dpo_comparison as dc\nprint('dpo-family src 就绪 (6 PO 变体统一框架)')"),
    md("## 1. 跑 6 变体的统一 benchmark"),
    code(MPL + """
hist = dc.benchmark(steps=50)
plt.figure(figsize=(8,4.5))
for name in hist:
    plt.plot(hist[name]['margin'], label=name.upper())
plt.xlabel('训练步'); plt.ylabel('偏好 margin (chosen − rejected)'); plt.legend(ncol=3)
plt.title('6 个偏好优化变体的训练动态 (统一框架)'); plt.tight_layout(); plt.show()
print('各变体最终 margin:')
for name in hist: print(f\"  {name.upper():7}: {hist[name]['margin'][-1]:+.3f}\")"""),
    md("""## 2. 反思
- DPO 家族都把「偏好对 (chosen > rejected)」变成可微损失, 不用单独 reward model + PPO。
- 变体差异在: 参考模型 (IPO 无 ref)、长度归一 (SimPO)、SFT 融合 (ORPO/CPO)、防 chosen 概率下降 (DPOP)。
- 本专题各 `*_minimal.py` 是每个变体的最小实现 (真实 gpt2 训练)。"""),
])

# 2. process-reward: PRM + Best-of-N
write("process-reward", "N1-prm-best-of-n.ipynb", [
    md("""# N1 · 过程奖励 (PRM) + Best-of-N 搜索

> 复用 `src/capstone_prm_bon.py` · 用 PRM (过程奖励模型) 给推理每一步打分, 配 Best-of-N 提升准确率。
> 对比 greedy / majority vote / BoN / weighted-BoN (mock GSM8K)。"""),
    code(PRE + "\nimport capstone_prm_bon as pb\nprint('process-reward src 就绪')"),
    md("## 1. 不同基础准确率下, 各推理时搜索策略的提升"),
    code(MPL + """
base_accs = [0.2, 0.3, 0.4, 0.5]
strategies = ['greedy','majority','bon','weighted_bon']
data = {s: [] for s in strategies}
for ba in base_accs:
    out = pb.evaluate_bon(100, 32, accuracy=ba)
    for s in strategies: data[s].append(out.get(s, 0))
plt.figure(figsize=(7.5,4.5))
for s in strategies: plt.plot(base_accs, data[s], 'o-', label=s)
plt.plot(base_accs, base_accs, 'k--', alpha=0.4, label='base (greedy 理论)')
plt.xlabel('基础 greedy 准确率'); plt.ylabel('策略准确率'); plt.legend()
plt.title('推理时搜索: PRM + Best-of-N 提升准确率 (mock GSM8K)'); plt.tight_layout(); plt.show()
print('base=0.3 时:', {s: round(data[s][1],3) for s in strategies})
print('→ majority/BoN 都提分; PRM 加权 BoN (按过程分选) 通常最好 (本专题核心)。')"""),
    md("> 本专题其余 src (`prm_minimal`/`math_shepherd_data_gen`/`mcts_llm`/`prime_minimal`/`rlvr_demo`/`bon_search`) 是 PRM 训练/搜索的实现。"),
])

# 3. reasoning-r1: GRPO countdown 训练曲线
write("reasoning-r1", "N1-grpo-countdown.ipynb", [
    md("""# N1 · R1-zero 式 GRPO 训练 (Countdown 任务)

> 复用 `src/r1_zero_track_a.py` + `grpo_minimal.py` · R1-zero 用 GRPO (组相对优势) + 可验证奖励 (RLVR)
> 在 Countdown (用数字凑目标) 上训练, 看 reward 随训练上升 (涌现推理)。(mock 动态)"""),
    code(PRE + "\nimport r1_zero_track_a as r1\nimport grpo_minimal as grpo\nimport random\nprint('reasoning-r1 src 就绪')"),
    md("## 1. GRPO 的核心: 组相对优势 (无需 critic)"),
    code("""# GRPO: 一个 prompt 采 k 个回答, 用组内相对 reward 当优势 (省掉 value model)
rewards = torch.tensor([0.2, 0.9, 0.1, 0.5, 0.8, 0.0, 0.7, 0.3])
adv = grpo.compute_group_advantage(rewards, k=8)
print('组内 8 个回答 reward:', rewards.tolist())
print('组相对优势 (减组均值/除组标准差):', [round(a,2) for a in adv.tolist()])
print('→ 高于组均值的回答得正优势 (被强化), 低于的负优势 (被抑制)。无需单独 critic (GRPO 省 value model)。')"""),
    md("## 2. Countdown 任务 + 可验证奖励 (RLVR) 训练曲线"),
    code(MPL + """
rng = random.Random(42)
nums, target = r1.gen_countdown_problem(rng)
print(f'Countdown 样例: 用 {nums} 凑出 {target}')
hist = r1.train_track_a(total_steps=200, k=8)
rew = [h['mean_reward'] for h in hist]
plt.figure(figsize=(7.5,4.2))
plt.plot(rew, alpha=0.5, label='每步 mean reward')
# 滑动平均
w=15; ma=np.convolve(rew, np.ones(w)/w, mode='valid')
plt.plot(range(w-1, len(rew)), ma, 'C3', lw=2, label=f'{w}步滑动平均')
plt.xlabel('GRPO 训练步'); plt.ylabel('mean reward (可验证: 答案对不对)'); plt.legend()
plt.title('R1-zero 式 GRPO: reward 随训练上升 (推理能力涌现)'); plt.tight_layout(); plt.show()
print(f'→ reward 从 {np.mean(rew[:20]):.2f} 升到 {np.mean(rew[-20:]):.2f} (RLVR: 用"答案可验证"当奖励, GRPO 优化)。')"""),
    md("> 本专题其余 src (`r1_zero_track_b`/`reinforce_pp`/`rloo_minimal`/`grpo_minimal`) 是 R1 复现的不同 RL 算法。"),
])

# 4. rl-sota-2026: DAPO 消融
write("rl-sota-2026", "N1-dapo-ablation.ipynb", [
    md("""# N1 · DAPO 四件套消融 (2026 RL SOTA)

> 复用 `src/capstone_dapo_ablation.py` · DAPO 在 GRPO 上加 4 个改进 (Clip-Higher / Dynamic Sampling /
> Token-level loss / Overlong reward)。看逐个加上去的累积提升 (mock)。"""),
    code(PRE + "\nimport capstone_dapo_ablation as da\nprint('rl-sota-2026 src 就绪')"),
    md("## 1. DAPO 四件套逐步消融"),
    code("""print('DAPO 4 件套累积消融 (从 GRPO baseline 起):')
da.run_ablation_grid()
print()
da.smoke_test_dapo_components()
print('\\n→ 每件套都加分, 全开 (DAPO full) 最好。这是 2026 RL SOTA 的工程组合 (本专题核心)。')"""),
    md("> 本专题其余 src (`dapo_minimal`/`dr_grpo`/`vapo_minimal`/`genrm`) 是 2026 RL SOTA 方法的最小实现。"),
])

# 5. rlhf-classic: reward hacking
write("rlhf-classic", "N1-reward-hacking.ipynb", [
    md("""# N1 · reward hacking: actor 钻 reward model 的空子

> 复用 `src/reward_hacking_demo.py` · RLHF 经典失败: reward model 把"长度"当质量代理,
> actor 学会"只加长不提质"来刷分。看这个 hacking 怎么发生、怎么检测 (接本专题讲义)。"""),
    code(PRE + "\nimport reward_hacking_demo as rh\nprint('rlhf-classic src 就绪')"),
    md("## 1. 模拟 actor 学到「加长→刷分」(质量不变)"),
    code(MPL + """
torch.manual_seed(0)
rewards, lens = [], []
for step in range(100):
    q = torch.randn(1)*0.1 + 0.5        # 质量基本不变
    L = torch.tensor([20.0 + step*0.5]) # actor 不断加长
    r = rh.hackable_reward(q, L, alpha=0.5)
    rewards.append(r.item()); lens.append(L.item())
diag = rh.detect_hacking(rewards, lens)
fig, ax = plt.subplots(figsize=(7.5,4.2))
ax.plot(rewards, 'C0-', label='reward (在涨!)')
ax.set_xlabel('训练步'); ax.set_ylabel('reward', color='C0')
ax2=ax.twinx(); ax2.plot(lens, 'C3--', label='回答长度 (在涨!)'); ax2.set_ylabel('长度', color='C3')
ax.set_title('reward hacking: reward 涨 = 质量提升? 不, 只是变长了'); plt.tight_layout(); plt.show()
print('hacking 检测:', diag)
print('→ reward 和长度高度相关而质量没变 = reward hacking (actor 钻 RM 的长度偏置)。')"""),
    md("""## 2. 修复 (本专题讲义)
- 降长度权重 α; 加 KL ref penalty (防漂出 SFT 分布); RM 训练加同长度对比削弱长度信号。
> 本专题其余 src (`rm_minimal`/`ppo_llm_minimal`/`sft_minimal`/`cai_minimal`) 是 RLHF 各环节最小实现。"""),
])

# 6. multimodal-agent: 毕业对比
write("multimodal-agent", "N1-rl-alignment-graduation.ipynb", [
    md("""# N1 · RL+对齐+推理系列毕业: 多方法对比

> 复用 `src/capstone_graduation.py` · 把 RL/对齐/推理/多模态/Agent 系列串起来, 在一个问题上对比不同方法。"""),
    code(PRE + "\nimport capstone_graduation as cg\nprint('multimodal-agent src 就绪')"),
    md("## 1. 同一问题, 不同方法 (SFT/RLHF/DPO/R1/...) 的对比"),
    code("""data = cg.export_for_notebook()
print('问题:', data.get('problem'))
print('标准答案:', data.get('ground_truth'))
print('\\n各方法结果:')
res = data.get('results', {})
if isinstance(res, dict):
    for method, r in res.items():
        print(f'  {method}: {r}')
elif isinstance(res, list):
    for r in res: print('  ', r)
print()
cg.print_capstone_comparison()"""),
    md("""## 2. 反思
- RL+对齐+推理系列: PEFT → RLHF → DPO家族 → 过程奖励 → R1推理 → SOTA → 多模态/Agent。
- 本专题 src (`s1_budget_forcing`/`vlm_r1_minimal`/`safe_rlhf_minimal`/`unified_view`) 串起多模态 + Agent + 安全 RL。"""),
])

print("done.")
