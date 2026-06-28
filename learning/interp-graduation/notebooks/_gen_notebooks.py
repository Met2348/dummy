"""生成 12.7 capstone notebooks (N1 完整interp流程 / N2 研究idea卡). 跑后 nbconvert --execute。"""
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
import interp_capstone as cap
cap.add_paths()
import numpy as np, torch"""

# ── N1 full-interp ──
n1 = new_notebook()
n1.cells = [
    md("""# N1 · 完整逆向工程: 对一个模型做一次 interp 研究

> 配套 12.7-L1 · capstone 的「把全套工具串成一次完整研究」:
> ① 装配检查 (M12 全套 src) ② 对玩具 transformer 跑**完整流程** (探针→patching→SAE),
> 产出一个连贯、因果、有证据的机制故事。"""),
    code(PATHS),
    md("## 1. 装配检查: M12 全套工具 import"),
    code("""for label, ok, detail in cap.assembly_check():
    print(f"  [{'OK ' if ok else 'FAIL'}] {label:26} {detail}")
print("\\n→ 探针/patching/SAE/circuits/CoT 全套工具就位 = 完整 interp 工具箱。")"""),
    md("## 2. 完整逆向工程流程 (探针→patching→SAE)"),
    code("""r = cap.run_full_interp(seed=0)
print(f"行为: 玩具 transformer 预测「下一个 = 当前 + 1」(任务准确率 {r['task_acc']:.2f})\\n")
print(f"② 探针 (12.2):    residual 线性编码「当前值」准确率 {r['probe_acc']:.2f}  [相关线索]")
print(f"③ patching (12.3): 答案信息因果定位在位置 {r['causal_pos']} (=最后位置 {r['last_pos']})  [因果证据]")
print(f"④ SAE (12.4):     「当前值」编码为单义特征, 纯度 {r['sae_purity']:.2f} >> 原始 {r['raw_purity']:.2f}  [表示]")"""),
    md("## 3. 连贯机制故事 (可视化证据链)"),
    code(MPL + """
fig, axes = plt.subplots(1, 3, figsize=(13,3.6))
axes[0].bar(['探针准确率'], [r['probe_acc']], color='C0'); axes[0].set_ylim(0,1.05)
axes[0].set_title('② 探针 (12.2)\\nresidual 编码「当前值」'); axes[0].text(0, r['probe_acc']+0.02, f"{r['probe_acc']:.2f}", ha='center')
axes[1].bar(['因果位置'], [r['causal_pos']], color='C2'); axes[1].axhline(r['last_pos'], ls='--', c='gray')
axes[1].set_title(f"③ patching (12.3)\\n因果定位=位置{r['causal_pos']}(最后)"); axes[1].set_ylim(0, r['last_pos']+1)
axes[2].bar(['原始神经元','SAE特征'], [r['raw_purity'], r['sae_purity']], color=['C3','C0']); axes[2].set_ylim(0,1)
axes[2].set_title('④ SAE (12.4)\\n单义纯度 SAE >> 原始')
plt.suptitle('完整逆向工程: 一条连贯、因果、有证据的机制证据链', fontsize=13); plt.tight_layout(); plt.show()
print('''机制故事 (capstone 产物):
  模型在「最后位置」的 residual 里 (③ patching 因果定位),
  用「单义特征」编码了「当前值」(② 探针读出 + ④ SAE 解叠加),
  然后 (经某组件) +1 得出预测。
  → 每一步都有证据 (探针=相关, patching=因果, SAE=表示), 不是讲故事。''')"""),
    md("""## 4. 反思
你做了一次**完整的逆向工程**。带走:
- **完整流程**: 选行为 → 探针 (读) → patching (因果) → SAE (表示) → 机制故事。
- **证据链**: 相关 (探针) + 因果 (patching) + 表示 (SAE) 多重验证, 不是单工具/讲故事。
- **严谨贯穿**: 相关vs因果 / 充要 / 分布内 / 可证伪 (M12.3-L4 + M9.3)。
> 真模型上同理 (你 M12 全程在 gpt2/TinyLlama 上做过 induction/logit lens/CoT)。
下一步 N2: 从「会解剖」到「能推进」— interp 研究 idea 卡。"""),
]
nbformat.write(n1, HERE / "N1-full-interp.ipynb")
print("written N1")

# ── N2 research-idea-cards ──
n2 = new_notebook()
n2.cells = [
    md("""# N2 · interp 研究 idea 卡 (重点 interp × reasoning)

> 配套 12.7-L2 · 用 M9 找 gap 框架, 在 interp 全套上产出 idea 卡。
> 挑 **interp × reasoning** (★, 你最可能转 PhD 题) 的, 细化成研究种子。"""),
    code(PATHS),
    md("## 1. interp 研究 gap 雷达 (★ = interp×reasoning 用户甜点)"),
    code("""print(cap.gap_radar())
print(f"\\n共 {len(cap.GAPS)} 个 gap; ★ 标记的是 interp×reasoning (接你 reasoning-r1 + NLP 背景)。")"""),
    md("## 2. 产出全部 idea 卡 (问题/为什么难/最小可验证实验/连接)"),
    code("""for g in cap.GAPS:
    print(cap.make_idea_card(g)); print()"""),
    md("## 3. 为你高亮: interp × reasoning 的 PhD 种子"),
    code(r"""mine = [g for g in cap.GAPS if '★' in g['area']]
print('对你 (NLP + reasoning + EE 数学) 量身定做的 gap:')
for g in mine:
    print(f"  ★ {g['area']}")
    print(f"     最小实验: {g['min_exp']}")
print(chr(10).join([
 '',
 '为什么是你的甜点 (L2):',
 '  - 地基全有: NLP/LLM(M1-8) + reasoning(reasoning-r1) + interp全套(M12) + 实验(M9)',
 '  - EE 数学优势: 线性代数(探针/QK-OV) + 因果干预 + 信号分析 = 研究生产力',
 '  - 缺口大 + 高stakes: interp×reasoning×安全是 frontier 实验室热点, 对 NLP 人友好',
 '  - toy 入口: 每个 gap 都配了小模型能跑的最小实验',
]))"""),
    md("""## 4. 反思 (12.7 收口 + Module 12 毕业)

你从「会用 interp 工具」走到「能在 interp 上找研究问题」。带走:
- **完整 interp 研究**: 探针→patching→SAE→circuit→CoT忠实, 连贯证据链 (N1)。
- **研究 gap**: 5 个 interp gap, 2 个 ★ (interp×reasoning) 对你量身定做。
- **你的 PhD 种子**: CoT 忠实性的机制级验证 / 计算vs陈述一致性 — 立刻能在小模型上手。

> **Module 12 毕业** 🎓: interp 全套 (probe/patch/SAE/circuits/CoT) 拿下; 能逆向工程 + 看穿诚实性。
> **能力网完成**: 会造模型 (M1-13) + 会理解模型 (M12) + 科研技能 (M9)。interp×reasoning 是你前沿研究的最佳切入。

**最后动手**: 把一个 ★ gap 的 idea 卡, 细化成你**明天能在小模型上跑**的具体步骤 (可证伪假设 + 最小实验)。这是你 PhD 研究的第一颗种子。"""),
]
nbformat.write(n2, HERE / "N2-research-idea-cards.ipynb")
print("written N2")
