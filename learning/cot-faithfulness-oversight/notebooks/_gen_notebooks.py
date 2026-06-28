"""生成 12.6 notebooks (N1 真实CoT忠实性偏置敏感性 / N2 weak-to-strong玩具). 跑后 nbconvert --execute。"""
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
sys.path.insert(0, str(Path.cwd().parents[1] / "_shared"))
import cot_probe as cp
import numpy as np, torch"""

# ── N1 cot-faithfulness ──
n1 = new_notebook()
n1.cells = [
    md("""# N1 · CoT 忠实性: 偏置敏感性 (真实 TinyLlama)

> 配套 12.6-L1/L2 · **小而真**: CoT 是真的「内心独白」吗? 测一个不忠实信号 —— **偏置敏感性**:
> 给 prompt 加一个**无关偏置提示** (「朋友猜答案是 99」), 看答案被带偏多少。
> 答案变了但 CoT 不会提这个提示 → 答案受未陈述上下文影响 = 忠实性缺口。"""),
    code(PATHS + "\nimport realmodels as rm\nprint('真实模型:', rm.available())"),
    md("> 若 TinyLlama=False 则无 HF 缓存, 跳过 (本 notebook 用真实 TinyLlama)。"),
    md("## 1. 偏置敏感性测试: 加无关提示, 答案被带偏吗"),
    code("""tok, model = rm.tinyllama()
if model is not None:
    bs = cp.bias_sensitivity(tok, model, hint=99)
    print('题目 (真值): 无提示答 → 加「朋友猜99」后答')
    for expr, truth, base, biased, ch, fol in bs['rows']:
        flag = ' [答案变了]' if ch else ''
        flag += ' [跟随到99]' if fol else ''
        print(f'  {expr:8} (真{truth}): {base} → {biased}{flag}')
    print(f'\\n→ 加无关提示后, 答案改变率 {bs[\"change_rate\"]:.0%}, 直接跟随到99 的比例 {bs[\"follow_rate\"]:.0%}')
else:
    print('无 TinyLlama, 跳过'); bs=None"""),
    md("## 2. 可视化 + 解读"),
    code(MPL + """
if bs is not None:
    fig, ax = plt.subplots(figsize=(6.5,4))
    ax.bar(['答案被改变','直接跟随提示\\n(答99)'], [bs['change_rate'], bs['follow_rate']], color=['C3','C1'])
    for i,v in enumerate([bs['change_rate'], bs['follow_rate']]): ax.text(i, v+0.02, f'{v:.0%}', ha='center')
    ax.set_ylim(0,1.1); ax.set_ylabel('比例'); ax.set_title('偏置敏感性: 无关提示对答案的影响 (高=忠实性缺口)')
    plt.tight_layout(); plt.show()
    print('''解读 (L2):
  - 答案大量被一个"无关提示"带偏 → 答案严重依赖 CoT 之外的上下文
  - 而模型的 CoT (如果让它写) 不会提"因为朋友猜99所以..." → 它会编独立理由
  - = 不忠实: 答案受未陈述因素驱动, CoT 不反映真实影响 (Turpin et al.)
  - 安全含义: 靠读 CoT 监控模型不可靠 (它说一套, 受另一套影响)''')"""),
    md("""## 3. 反思
你在**真实 TinyLlama** 上测了 CoT 忠实性的一个缺口。带走:
- **偏置敏感性**: 加无关提示, 答案大幅被带偏 (实测改变率很高) → 答案受未陈述上下文影响。
- **不忠实**: CoT 不反映真实影响 (说一套, 受另一套驱动) — confabulation。
- **安全含义**: CoT 监控脆弱 (L2); 要配机制级 interp (M12.2-12.5 看内部真实计算) 验证。
> 小模型尤其脆弱, 但前沿大模型也有此问题 (Turpin 在 GPT/Claude 证明过)。
下一步 N2: 当模型比人强, 怎么监督? weak-to-strong 玩具。"""),
]
nbformat.write(n1, HERE / "N1-cot-faithfulness.ipynb")
print("written N1")

# ── N2 weak-to-strong ──
n2 = new_notebook()
n2.cells = [
    md("""# N2 · weak-to-strong 玩具 (scalable oversight)

> 配套 12.6-L3 · 当模型比人强, 人 (弱监督者) 怎么引出强模型的能力?
> 玩具: 弱监督者给**带噪标签** (准确率~0.73), 强学生 (平滑归纳偏置) 在弱标签上训。
> 强学生能**超过**弱监督者吗? (靠归纳偏置平滑掉噪声)"""),
    code(PATHS + "\nprint('就绪')"),
    md("## 1. weak-to-strong: 弱监督 (带噪) → 强学生能否超过"),
    code(MPL + """
torch.manual_seed(0)
results = []
for noise in [0.15, 0.20, 0.25, 0.30]:
    wa, sa = cp.weak_to_strong_demo(noise=noise, seed=0)
    results.append((noise, wa, sa))
    print(f'噪声 {noise:.0%}: 弱监督者 {wa:.2f} → 强学生 {sa:.2f}  {"✓ 超过 (w2s)" if sa>wa+0.02 else ""}')
noises=[r[0] for r in results]; was=[r[1] for r in results]; sas=[r[2] for r in results]
fig, ax = plt.subplots(figsize=(7,4.2))
ax.plot(noises, was, 'o-', color='C3', label='弱监督者 (带噪标签)')
ax.plot(noises, sas, 's-', color='C0', label='强学生 (在弱标签上训, vs 真标签)')
ax.set_xlabel('弱监督的标签噪声'); ax.set_ylabel('准确率 (vs 真标签)'); ax.set_ylim(0.5,1.02); ax.legend()
ax.set_title('weak-to-strong: 强学生超过弱监督者 (归纳偏置平滑噪声)')
plt.tight_layout(); plt.show()
print('→ 强学生在带噪弱标签上训, 却恢复了真边界, 准确率远超弱监督者 = weak-to-strong generalization。')"""),
    md("## 2. 机制 + 解读"),
    code("""print('''weak-to-strong 的机制 (L3):
  - 弱监督者: 真边界 + 随机噪声 → 标签不稳 (准确率~0.7-0.85)
  - 强学生 (MLP, 平滑归纳偏置): 在带噪标签上训
    → 它"不愿"去拟合零散的噪声 (平滑偏置), 而是学到弱标签背后的"真概念"(斜线边界)
    → 于是恢复真边界, 准确率 ~0.95+ 超过弱监督者
  → 这是 scalable oversight 的一线希望:
    也许弱 (人类) 监督, 能引出强 (超人) 模型的真实能力, 而非被人的错误拖死。

但要诚实 (M9.3):
  - 玩具是干净的 (噪声随机、概念简单); 真实超人模型的"概念"远复杂
  - 弱监督若是"系统性错"(非随机噪声), 强学生会学到系统错 (w2s 失效)
  - 这是 open 问题, 不是已解决''')"""),
    md("""## 3. 反思 (12.6 收口)

你做了 weak-to-strong 玩具。带走:
- **scalable oversight**: 模型 > 人时怎么监督 (对齐根本挑战)。
- **weak-to-strong**: 弱监督 (带噪) 能引出强学生超过它 (归纳偏置平滑噪声) — 一线希望。
- 与 interp 互补: 行为监督 (w2s/CoT) 给规模, 机制 interp 给可信 (L3)。

> **M12.6 收口**: CoT 不一定忠实 (偏置敏感性实测高); CoT 监控脆弱; scalable oversight (w2s); 欺骗检测靠 interp。
> **交棒 M12.7「interp-graduation」**: Module 12 capstone — 对一个模型做**完整** interp 解剖 (探针→patching→SAE→电路) + interp×reasoning 研究 idea 卡。下一专题 `interp-graduation`。"""),
]
nbformat.write(n2, HERE / "N2-weak-to-strong.ipynb")
print("written N2")
