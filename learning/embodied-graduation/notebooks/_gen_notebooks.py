"""生成 11.7 capstone notebooks (N1 装配+mini-benchmark / N2 研究idea卡). 跑后 nbconvert --execute。"""
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
import embodied_capstone as cap
cap.add_paths()   # 把 M11 全栈 src 加进 path
import numpy as np, torch"""

# ── N1 assemble-and-benchmark ──
n1 = new_notebook()
n1.cells = [
    md("""# N1 · 装配检查 + mini-benchmark

> 配套 11.7-L1/L3 · capstone 的「亲手把全栈跑一遍」:
> ① **装配检查**: 一次 import M11 全栈 src, 证明组合成一个栈。
> ② **mini-benchmark**: 多种方法 (BC / mini-VLA / 世界模型MPC) 在同一 toy 控制任务比成功率
> (LIBERO/CALVIN 思路: 标准任务 + 成功率 + 可比)。"""),
    code(PATHS),
    md("## 1. 装配检查: M11 全栈 import + 最小烟测"),
    code("""for label, ok, detail in cap.assembly_check():
    print(f"  [{'OK ' if ok else 'FAIL'}] {label:24} {detail}")
print("\\n→ 全 7 个 src 都能 import + 跑通烟测 = 你掌握了完整具身栈。")"""),
    md("## 2. mini-benchmark: 多方法在同一任务比成功率 (标准化评测)"),
    code("""torch.manual_seed(0)
import toy_env as env, mini_vla as vla, bc_train as bc, world_model as wm
S, A = env.make_demos(n=300, seed=0)
results = {}

# 方法1: BC (M11.4)
mbc = bc.build_bc_policy(); bc.train_bc(mbc, S, A, epochs=300)
results['BC (M11.4)'] = env.eval_policy(bc.bc_policy_fn(mbc), n_episodes=200)

# 方法2/3: mini-VLA 离散/连续 (M11.2)
for head in ['discrete', 'continuous']:
    m = vla.build_mini_vla(head=head); vla.train_vla(m, S, A, epochs=300)
    results[f'mini-VLA {head} (M11.2)'] = env.eval_policy(vla.make_policy(m), n_episodes=200)

# 方法4: 世界模型 + MPC (M11.5, 零专家! 用随机数据)
Sr, Ar, Dr = wm.make_random_transitions(n=3000, seed=0)
wmodel = wm.build_world_model(); wm.train_world_model(wmodel, Sr, Ar, Dr, epochs=400)
results['世界模型+MPC (M11.5)'] = env.eval_policy(wm.mpc_policy_fn(wmodel, n_samples=150, horizon=6), n_episodes=100)

print('mini-benchmark (2D 到达任务成功率):')
for k, v in results.items():
    print(f'  {k:26}: {v:.2f}')"""),
    md("## 3. 可视化对比 (像 benchmark 排行榜)"),
    code(MPL + """
names = list(results.keys()); vals = [results[k] for k in names]
fig, ax = plt.subplots(figsize=(9,4.2))
colors = ['C0','C1','C2','C3']
ax.barh(names, vals, color=colors[:len(names)])
for i,v in enumerate(vals): ax.text(v+0.01, i, f'{v:.2f}', va='center')
ax.set_xlim(0,1.1); ax.set_xlabel('成功率'); ax.set_title('mini-benchmark: M11 各方法在 2D 到达任务')
plt.tight_layout(); plt.show()
print('解读 (接 L3 评测):')
print('  - BC / mini-VLA: 用专家 demo 监督学, 成功率高')
print('  - 世界模型+MPC: 零专家 (随机数据) + 规划, 也能解 — 数据来源不同 (M11.5)')
print('  - 成功率可比, 但要记: 它抓不全安全/平滑/泛化 (L3 评测的坑)')"""),
    md("""## 4. 反思
你跑了 M11 全栈 + 一个 mini-benchmark。带走:
- **装配**: 7 个 src 组合成完整具身栈 — 你掌握的是体系不是碎片。
- **多方法可比**: BC / mini-VLA / 世界模型MPC 在同一任务比成功率 (LIBERO/CALVIN 思路)。
- **评测要深**: 成功率必要不充分; 真评要看分层泛化 + 鲁棒性 + sim2real (L3)。
下一步 N2: 从「会用」到「能推进」— 产出具身研究 idea 卡。"""),
]
nbformat.write(n1, HERE / "N1-assemble-and-benchmark.ipynb")
print("written N1")

# ── N2 research-idea-cards ──
n2 = new_notebook()
n2.cells = [
    md("""# N2 · 具身研究 idea 卡

> 配套 11.7-L3 · 用 M9 找 gap 框架, 在具身全栈产出可执行 idea 卡。
> 挑最匹配你 (NLP 背景 + IsaacLab 经验) 的, 细化成研究种子。"""),
    code(PATHS),
    md("## 1. 具身/VLA 研究 gap 雷达 (每个标连接哪些专题)"),
    code("""print(cap.gap_radar())
print(f"\\n共 {len(cap.GAPS)} 个 gap。好 gap 长在两个你都懂的领域交界。")"""),
    md("## 2. 产出全部 idea 卡 (问题/为什么难/最小可验证实验/连接)"),
    code("""for g in cap.GAPS:
    print(cap.make_idea_card(g)); print()"""),
    md("## 3. 为你 (NLP 博 0 + IsaacLab) 高亮最匹配的 gap"),
    code(r"""mine = [g for g in cap.GAPS if any(k in g['connects'] for k in ['数据','sim2real','DR','M11.6','M13.6'])]
print('对你最匹配的 gap (低地基成本 + 你的 IsaacLab/数据优势):')
for g in mine:
    print(f"  * {g['area']}  <-  {g['connects']}")
print(chr(10).join([
 '',
 '挑选准则 (接 M9 研究品味):',
 '  - 低地基成本: 数据高效具身 接你 NLP 数据/scaling 直觉',
 '  - 独特优势: sim2real/DR 接你真实 IsaacLab 经验 (稀缺手感)',
 '  - 独有交叉: dLLM(M13.6) 双向可控生成 用于动作生成 = 你独有',
]))"""),
    md("""## 4. 反思 (11.7 收口 + Module 11 毕业)

你从「会用全栈」走到「能在全栈上找研究问题」。带走:
- **找 gap 可操作**: 6 个具身 gap, 每个带最小可验证实验 (toy 上能跑第一步)。
- **对你**: 数据高效具身 (接数据直觉) + sim2real/DR (接 IsaacLab 经验) 最匹配; dLLM×具身是独有交叉。
- **元能力**: 拆解 (L2) + 找 gap (L3) + 复用直觉, 跨模块通用。

> **Module 11 毕业** 🎓: 具身全栈 (动作表示→架构→动作头→数据→世界模型→sim2real) 拿下。
> **下一站**: M12 机制可解释性 (打开你造的 VLA/LLM 看内部) — 理解你造的所有模型的钥匙。

**最后动手**: 从高亮的 gap 挑一个, 把 idea 卡的「最小可验证实验」细化成你**明天能在本课 toy 上跑**的具体步骤。这是你具身研究的第一颗种子。"""),
]
nbformat.write(n2, HERE / "N2-research-idea-cards.ipynb")
print("written N2")
