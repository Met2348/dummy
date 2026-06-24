"""生成 13.6 notebooks (N1 masked diffusion 并行解码 / N2 dLLM vs AR). 跑后 nbconvert --execute。"""
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

# ── N1 masked-diffusion-lm ──
n1 = new_notebook()
n1.cells = [
    md("""# N1 · masked diffusion 语言模型: 并行迭代解码

> 配套 13.6-L1 · 文本也能扩散。玩具「语言」= 回文序列 [a,b,c,c,b,a]。
> 手搭一个 masked diffusion LM (dLLM): 从全 [MASK] 出发, **按置信度并行迭代解码**生成文本。
> 看非自回归生成长什么样, 以及「并行度 vs 质量」的权衡 (呼应 M13.2 采样步数)。"""),
    code(PATHS + "\nimport numpy as np, torch\nimport diffusion_lm as dl\nprint('diffusion_lm 就绪; 词表', dl.V, '长度', dl.L, '[MASK]id', dl.MASK)"),
    md("## 1. 玩具语言: 回文 (位置 i 必须等于位置 L-1-i)"),
    code("""data = dl.make_sequences(2000, seed=0)
print('回文数据', data.shape, '合法率', dl.is_palindrome(data))
for s in data[:5]: print('  ', s)"""),
    md("## 2. 训练 dLLM (随机比例遮盖, 只在被遮位预测原 token = masked diffusion)"),
    code("""torch.manual_seed(0)
dlm = dl.build_dlm(d_model=80)
losses = dl.train_dlm(dlm, data, epochs=800)
print(f'dLLM 训练: loss {losses[0]:.3f} → {losses[-1]:.3f}')"""),
    md("## 3. 看一条序列的并行解码轨迹 (从全 [MASK] 逐轮填充)"),
    code(MPL + """
gen, traj = dl.generate_dlm(dlm, n=200, rounds=3, seed=1, record=True)
print('解码轨迹 (6=[MASK]):')
for i, s in enumerate(traj):
    print(f'  轮{i}: {s}')
# 可视化: 每轮哪些位被填上
fig, ax = plt.subplots(figsize=(7,3.2))
arr = np.array(traj)
import numpy as np
show = np.where(arr==dl.MASK, -1, arr)
im = ax.imshow(show, cmap='tab10', vmin=-1, vmax=dl.V, aspect='auto')
ax.set_yticks(range(len(traj))); ax.set_yticklabels([f'轮{i}' for i in range(len(traj))])
ax.set_xlabel('序列位置'); ax.set_title('dLLM 并行迭代解码: 每轮提交最自信的若干位 (灰=仍 MASK)')
for i in range(len(traj)):
    for j in range(dl.L):
        v = arr[i,j]
        ax.text(j, i, 'M' if v==dl.MASK else str(v), ha='center', va='center', fontsize=9,
                color='gray' if v==dl.MASK else 'white')
plt.tight_layout(); plt.show()
print('→ 不是左到右一个个生成, 而是"哪里有把握先填哪里", 多位并行 (非自回归)。')"""),
    md("## 4. 并行度 vs 质量: 解码轮数的权衡 (呼应 M13.2 采样步数)"),
    code(MPL + """
rounds_list = [1,2,3,4,6]
valid = [dl.is_palindrome(dl.generate_dlm(dlm, n=300, rounds=r, seed=1)) for r in rounds_list]
parallel = [dl.L/r for r in rounds_list]
fig, ax1 = plt.subplots(figsize=(7,4))
ax1.plot(rounds_list, valid, 'o-', color='C0', label='合法率')
ax1.set_xlabel('解码轮数'); ax1.set_ylabel('合法回文率', color='C0'); ax1.set_ylim(0,1.05)
ax2 = ax1.twinx(); ax2.plot(rounds_list, parallel, 's--', color='C3', label='并行度')
ax2.set_ylabel('并行度 (L/轮数)', color='C3')
plt.title('dLLM: 轮数越多质量越高但越不并行 (同扩散 步数vs质量, M13.2)')
fig.tight_layout(); plt.show()
print('合法率:', {r:round(v,2) for r,v in zip(rounds_list,valid)})
print('→ 轮数=1 全并行但质量低; 轮数=L 退化为顺序解码但质量满分。这就是 dLLM 的核心旋钮。')"""),
    md("""## 5. 反思
你手搭了一个 dLLM 并看它**并行解码**生成文本。带走:
- **masked diffusion**: 前向逐步遮盖 token, 模型学「填被遮位」; 生成 = 从全 [MASK] 按置信度迭代填。
- **非自回归**: 不是左到右一个个吐, 而是「有把握先填」, 多位并行 (区别于 AR 的本质)。
- **并行度 vs 质量旋钮**: 轮数少更并行但质量低, 轮数多更准但更慢 —— 和扩散的步数vs质量 (M13.2) 同构。
下一步 N2: dLLM 比 AR 强在哪? 看**双向 infilling** (dLLM 杀手锏)。"""),
]
nbformat.write(n1, HERE / "N1-masked-diffusion-lm.ipynb")
print("written N1")

# ── N2 dllm-vs-ar ──
n2 = new_notebook()
n2.cells = [
    md("""# N2 · dLLM vs 自回归 (AR): 双向 infilling 的胜利

> 配套 13.6-L2/L3 · 同一份回文数据, 训 dLLM 和 AR 两个模型。
> 对比: ① 都能生成合法序列, 但范式不同 ② **双向 infilling** —— 挖掉中间靠左的位 (镜像在右侧),
> dLLM 看双侧→准, AR 因果只看左侧→瞎猜。这是 dLLM 对 NLP 的核心价值。(接 9.4 对照实验)"""),
    code(PATHS + "\nimport numpy as np, torch\nimport diffusion_lm as dl\nprint('就绪')"),
    md("## 1. 同数据训 dLLM 和 AR"),
    code("""torch.manual_seed(0)
data = dl.make_sequences(2000, seed=0)
dlm = dl.build_dlm(d_model=80); dl.train_dlm(dlm, data, epochs=800)
ar  = dl.build_ar(d_model=80);  dl.train_ar(ar, data, epochs=800)
gen_dlm = dl.generate_dlm(dlm, n=300, rounds=dl.L, seed=1)   # 足够轮数保质量
gen_ar  = dl.generate_ar(ar, n=300, seed=1)
print(f'dLLM 生成 (并行迭代, {dl.L}轮): 合法回文率 {dl.is_palindrome(gen_dlm):.2f}')
print(f'AR   生成 (左到右, {dl.L}步):   合法回文率 {dl.is_palindrome(gen_ar):.2f}')
print('→ 两者都能学会生成合法序列; 区别在"怎么生成"和"能不能双向填空"。')"""),
    md("""## 2. 杀手锏: 双向 infilling
挖掉**位置 1** (它的镜像在位置 4, 在右侧)。要填对, 必须看**右侧**上下文:
- dLLM 双向 attention → 看得到位置 4 → 填对
- AR 因果 attention → 位置 1 只能看位置 0 → 看不到右侧 → 瞎猜"""),
    code(MPL + """
test = dl.make_sequences(400, seed=9)
accs, labels = [], []
for bp in [1, 2]:     # 两个"靠左、镜像在右"的位置
    a_dlm = dl.dlm_infill_accuracy(dlm, test, bp)
    a_ar  = dl.ar_infill_accuracy(ar, test, bp)
    accs.append((a_dlm, a_ar)); labels.append(f'位置{bp}\\n(镜像在位置{dl.L-1-bp})')
import numpy as np
x = np.arange(len(labels)); w=0.35
fig, ax = plt.subplots(figsize=(7,4))
ax.bar(x-w/2, [a[0] for a in accs], w, label='dLLM (双向)', color='C0')
ax.bar(x+w/2, [a[1] for a in accs], w, label='AR (单向)', color='C3')
ax.axhline(1/dl.V, ls=':', color='gray', label=f'瞎猜 1/V={1/dl.V:.2f}')
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('infill 准确率'); ax.set_ylim(0,1.1); ax.legend()
ax.set_title('双向 infilling: dLLM 看右文→准, AR 看不到→瞎猜')
plt.tight_layout(); plt.show()
for (ad,aa),lb in zip(accs,labels):
    print(f'{lb.split(chr(10))[0]}: dLLM {ad:.2f}  AR {aa:.2f}')
print('→ dLLM 双向上下文 = 天生会"完形填空"; AR 单向看不到右文。这是 dLLM 对可控/编辑类任务的核心优势。')"""),
    md("## 3. AR vs dLLM 范式对照表"),
    code("""print('''
┌─────────────┬──────────────────────┬──────────────────────────┐
│             │ 自回归 AR (主流 LLM)  │ 扩散 dLLM (LLaDA 等)      │
├─────────────┼──────────────────────┼──────────────────────────┤
│ 生成顺序    │ 左到右, 一次一个      │ 任意位, 按置信度并行迭代  │
│ 注意力      │ 因果(单向)            │ 双向                      │
│ 步数        │ = 序列长度 L          │ T 轮 (可 < L, 质量换并行) │
│ infilling   │ 弱(只看左文)          │ 强(看左右全文)            │
│ 可控/编辑   │ 较难                  │ 较易(双向+任意位)         │
│ 成熟度      │ 极成熟(生态/工具全)   │ 新兴(2025-2026 热点)      │
└─────────────┴──────────────────────┴──────────────────────────┘''')"""),
    md("""## 4. 反思 (13.6 收口)

你在同一份数据上对比了 dLLM 和 AR, 看清了 dLLM 对 NLP 的价值。带走:
- **范式差异**: AR 左到右因果; dLLM 任意位并行 + 双向。机制都是 transformer, 只换「生成范式」。
- **dLLM 杀手锏**: 双向 infilling / 可控编辑 (本 notebook 实测 dLLM 准、AR 瞎猜)。
- **代价**: 并行度 vs 质量旋钮 (N1); 生态远不如 AR 成熟 (开放问题, 用 M9.3 批判读)。
- 为什么你 NLP 人该关注: dLLM 是 AR 之外的另一条生成范式, 在并行解码/可控生成上有结构性优势。

> **M13.6 收口**: 生成式媒体绕一圈回 NLP —— 文本也能扩散, dLLM = transformer + 离散扩散范式。
> **交棒 M13.7「generative-media-graduation」**: capstone, 把 M13 全链 (扩散→流→DiT→视频→世界模型→dLLM) 装配起来 + 找研究 gap。下一专题 `generative-media-graduation`。"""),
]
nbformat.write(n2, HERE / "N2-dllm-vs-ar.ipynb")
print("written N2")
