"""生成 13.7 capstone notebooks (N1 装配+统一生成 / N2 研究 idea 卡). 跑后 nbconvert --execute。"""
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
import generative_capstone as gc
gc.add_paths()   # 把 M13 全 6 专题的 src 加进 path"""

# ── N1 assemble-and-generate ──
n1 = new_notebook()
n1.cells = [
    md("""# N1 · 装配检查 + 统一生成画廊

> 配套 13.7-L1/L2 · capstone 的「亲手把全链跑一遍」:
> ① **装配检查**: 一次 import M13 全 6 专题的 src, 证明它们组合成一个栈 (你真的能驱动每一种生成方法)。
> ② **统一生成画廊**: 一口气驱动扩散/流匹配/DiT/dLLM 四种生成, 同屏对比。"""),
    code(PATHS),
    md("## 1. 装配检查: M13 全链 import + 最小烟测 (跨专题 src 复用)"),
    code("""for topic, ok, detail in gc.assembly_check():
    print(f"  [{'OK ' if ok else 'FAIL'}] {topic:22} {detail}")
print("\\n→ 全 6 专题的 src 都能 import + 跑通最小烟测 = 你掌握了完整的生成式媒体栈。")"""),
    md("## 2. 统一生成画廊: 一口气驱动四种生成方法"),
    code(MPL + """
import numpy as np, torch
import diffusion as d, flow_matching as fm, dit, diffusion_lm as dl
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# (a) 13.1 扩散: 2D 双月
torch.manual_seed(0)
x0 = d.make_two_moons(400, seed=0)
m = d.build_denoiser(); _, sched = d.train_diffusion(m, x0, epochs=300)
gen = d.sample(m, sched, n=400, dim=2, seed=1)
axes[0].scatter(gen[:,0], gen[:,1], s=6, alpha=0.5, color='C0')
axes[0].set_title('13.1 扩散\\n(生成双月分布)'); axes[0].axis('equal')

# (b) 13.2 流匹配: 少步采样
torch.manual_seed(0)
x1 = fm.make_two_moons(400, seed=0)
vf = fm.build_velocity_field(); fm.train_flow_matching(vf, x1, epochs=300)
genf = fm.sample(vf, n=400, dim=2, steps=8, seed=1)
axes[1].scatter(genf[:,0], genf[:,1], s=6, alpha=0.5, color='C2')
axes[1].set_title('13.2 流匹配\\n(8 步采样)'); axes[1].axis('equal')

# (c) 13.3 DiT: 条件生成 (4 类)
torch.manual_seed(0)
xb, yb = dit.make_class_blobs(n_per=150, seed=0)
dm = dit.build_dit(); _, dsched = dit.train_dit(dm, xb, yb, epochs=400)
for c in range(4):
    s = dit.sample(dm, dsched, cls=c, n=80, guidance=2.0, seed=3)
    axes[2].scatter(s[:,0], s[:,1], s=6, alpha=0.5, color=f'C{c}')
axes[2].set_title('13.3 DiT\\n(条件生成 4 类)'); axes[2].axis('equal')

# (d) 13.6 dLLM: 文本(回文)生成
torch.manual_seed(0)
data = dl.make_sequences(1500, seed=0)
dlm = dl.build_dlm(d_model=80); dl.train_dlm(dlm, data, epochs=500)
gens = dl.generate_dlm(dlm, n=200, rounds=dl.L, seed=1)
valid = dl.is_palindrome(gens)
axes[3].axis('off'); axes[3].set_title(f'13.6 dLLM\\n(回文生成 合法率 {valid:.2f})')
for i, s in enumerate(gens[:8]):
    axes[3].text(0.5, 0.9-i*0.11, str(s), ha='center', fontsize=11, family='monospace')
plt.suptitle('统一生成画廊: 一口气驱动 M13 四种生成方法 (扩散/流/DiT/dLLM)', fontsize=13)
plt.tight_layout(); plt.show()
print(f'→ 扩散(双月) / 流匹配(8步) / DiT(条件4类) / dLLM(回文 {valid:.2f}) 全部由你驱动。')"""),
    md("""## 3. 反思
你亲手把 M13 全链跑了一遍。带走:
- **装配检查**: 6 个专题的 src 组合成一个栈 —— 你掌握的是完整生成式媒体栈, 不是碎片。
- **统一生成**: 同一套「生成=预测」信念下, 四种方法 (连续扩散/流/条件 DiT/离散 dLLM) 全跑通。
- 真实系统 (SD/Sora) = 这些部件 + 规模 (L2 已拆解), 你懂每一块。
下一步 N2: 从「会用」到「能推进」—— 产出 M13 的研究 idea 卡。"""),
]
nbformat.write(n1, HERE / "N1-assemble-and-generate.ipynb")
print("written N1")

# ── N2 research-idea-cards ──
n2 = new_notebook()
n2.cells = [
    md("""# N2 · 研究 idea 卡: 从「会用」到「能推进」

> 配套 13.7-L3 · 用 M9 的找 gap 框架, 在 M13 全链上产出可执行的研究 idea 卡。
> 然后挑一张最适合你 (NLP 背景) 的, 细化成自己的研究种子。"""),
    code(PATHS),
    md("## 1. M13 研究 gap 雷达 (7 个, 每个标注连接哪些专题)"),
    code("""print(gc.gap_radar())
print(f"\\n共 {len(gc.GAPS)} 个 gap。好 gap 长在「两个你都懂的领域」的交界。")"""),
    md("## 2. 产出全部 idea 卡 (问题 / 为什么难 / 最小可验证实验 / 连接)"),
    code("""for g in gc.GAPS:
    print(gc.make_idea_card(g)); print()"""),
    md("## 3. 为你 (NLP 博 0) 高亮最匹配的 gap"),
    code("""# 挑「连接里含 dLLM 或 NLP」的 gap = 你地基成本最低、交界优势最大的
mine = [g for g in gc.GAPS if 'dLLM' in g['connects'] or 'DPO' in g['connects']]
print('对你 (NLP/LLM 背景) 最匹配的 gap (低地基成本 + 交界优势):')
for g in mine:
    print(f"  ★ {g['area']}  ←  {g['connects']}")
print('''
挑选准则 (接 M9 研究品味):
  - 低地基成本: 你已有大半地基 (transformer + 扩散 + 对齐)
  - toy 入口:   本课每个 gap 都配了能在小规模跑的最小实验
  - 交界优势:   你懂 NLP + 扩散, dLLM 范式迁移你比别人多懂一边
  - 赔率:       dLLM 够新有红利, 已证明可 scale 不是空中楼阁''')"""),
    md("""## 4. 反思 (13.7 收口 + Module 13 毕业)

你从「会用全栈」走到了「能在全栈上找研究问题」。带走:
- **找 gap 是可操作的**: 四来源 (拆解残留/专题标注/跨专题迁移/评估滞后), 范式迁移是富矿。
- **idea 卡四要素**: 问题 / 为什么难 / **最小可验证实验** / 连接 —— 重点是「明天能跑的第一步」。
- **对你**: dLLM 相关 gap (1/4/5) 地基成本最低、赔率最高 —— 你的 transformer+扩散+对齐知识全用得上。

> **Module 13 毕业** 🎓: 扩散→流→DiT→视频→世界模型→dLLM 全链拿下; 三个元能力 (拆解/找gap/迁移直觉) 跨模块通用。
> **下一站**: M11 具身 (世界模型落地机器人, 共享 `world_model.py`) 或 M12 可解释 (打开你造的生成模型看内部)。两条路都和 M13 共享地基。

**最后动手**: 从高亮的 dLLM gap 里挑一个, 把 idea 卡的「最小可验证实验」细化成你**明天就能在本课 toy 上跑**的具体步骤。这是你研究路线的第一颗种子。"""),
]
nbformat.write(n2, HERE / "N2-research-idea-cards.ipynb")
print("written N2")
