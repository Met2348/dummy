"""生成 12.5 notebooks (N1 真实gpt2找induction head / N2 逐头贡献热图). 跑后 nbconvert --execute。"""
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
import circuits as ci
import realmodels as rm
import numpy as np, torch
print('真实模型:', rm.available())"""

# ── N1 find-induction-head ──
n1 = new_notebook()
n1.cells = [
    md("""# N1 · 在真实 gpt2 找 induction head

> 配套 12.5-L2 · **小而真**: induction head 是 mech interp 最著名的 circuit (in-context learning 机制)。
> 喂「随机 token 重复两遍」, 逐头算 induction 分数, 在真 gpt2 找到它, 看注意力图, 并消融验证因果。"""),
    code(PATHS),
    md("> 若 gpt2=False 则无 HF 缓存, 跳过 (本 notebook 全程真实 gpt2)。"),
    md("## 1. 喂重复序列, 逐 (层,头) 算 induction 分数"),
    code(MPL + """
tok, model = rm.gpt2(output_attentions=True)
if model is not None:
    tokens, k = ci.make_repeated_tokens(tok, n_unique=24, seed=0)
    print(f'序列: {tokens.shape[1]} token (BOS + {k} 随机 token 重复两遍)')
    scores = ci.induction_scores(model, tokens, k)
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(scores, cmap='viridis', aspect='auto')
    ax.set_xlabel('头 (head)'); ax.set_ylabel('层 (layer)'); ax.set_title('gpt2 各头的 induction 分数\\n(亮=attend到「上次出现的下一个token」=induction)')
    plt.colorbar(im); plt.tight_layout(); plt.show()
    best = np.unravel_index(np.argmax(scores), scores.shape)
    print(f'→ induction 分数最高: 层{best[0]} 头{best[1]} (分数 {scores[best]:.2f}) = gpt2 的一个 induction head!')
else:
    print('无 gpt2')"""),
    md("## 2. 看这个头的注意力图 (它真的 attend 到「上次的下一个」吗)"),
    code(MPL + """
if model is not None:
    with torch.no_grad():
        out = model(tokens, output_attentions=True)
    a = out.attentions[best[0]][0, best[1]].numpy()   # (seq, seq) 该头的注意力
    fig, ax = plt.subplots(figsize=(6.5,5.5))
    im = ax.imshow(a, cmap='magma'); ax.set_xlabel('被 attend 的位置 (key)'); ax.set_ylabel('query 位置')
    ax.set_title(f'层{best[0]}头{best[1]} 注意力图\\n第二遍(下半)出现"偏移对角线"=induction 复制')
    # 标出 induction 对角线 (第二遍 i → i-k+1)
    xs = [(i-k)+1 for i in range(1+k, tokens.shape[1])]; ys = list(range(1+k, tokens.shape[1]))
    ax.plot(xs, ys, 'c+', ms=6, alpha=0.7, label='induction 目标')
    ax.legend(); plt.colorbar(im); plt.tight_layout(); plt.show()
    print('→ 第二遍每个位置的注意力落在「第一遍该token的下一个」(青十字) → induction 模式坐实。')"""),
    md("## 3. 因果验证: 消融这个头, 预测重复变差吗"),
    code("""if model is not None:
    base = ci.induction_loss(model, tokens, k)
    abl = ci.induction_loss(model, tokens, k, ablate=tuple(best))
    print(f'induction loss (预测重复的交叉熵, 越低越好):')
    print(f'  完整模型:        {base:.3f}')
    print(f'  消融 层{best[0]}头{best[1]}: {abl:.3f}  ({"变差 ✓ 因果确认" if abl>base else "没变差"})')
    print(f'  损害: {abl-base:+.3f}')
    print('→ 消融 induction head 使"预测重复"变差 = 因果确认它负责 induction (M12.3 干预)。')
    print('  (损害可能不大, 因为 gpt2 有多个 induction head 冗余 — N2 会看到整组)')"""),
    md("""## 4. 反思
你在**真实 gpt2** 上找到并验证了 induction head。带走:
- **induction head**: attend 到「上次出现的下一个 token」→ 复制 = in-context learning 机制 (L2)。
- **真实大模型的真实 circuit**: gpt2 预训练涌现, 你用注意力分数找到它、消融因果验证。
- **冗余**: 损害不大 = 有备份头 (L3); N2 看整组。
下一步 N2: 逐头贡献热图, 看 induction 是一组头协作。"""),
]
nbformat.write(n1, HERE / "N1-find-induction-head.ipynb")
print("written N1")

# ── N2 head-attribution ──
n2 = new_notebook()
n2.cells = [
    md("""# N2 · 逐头贡献热图 (induction circuit 的组件分布)

> 配套 12.5-L3/L4 · 逐头消融, 测每个 head 对 induction 的贡献 (消融它 induction loss 涨多少)。
> 看 induction 是**一组头协作** (冗余), 不是单个 — 这解释 N1 单头消融损害小。"""),
    code(PATHS),
    md("## 1. 逐头消融贡献热图 (144 个 head, 稍慢)"),
    code(MPL + """
tok, model = rm.gpt2(output_attentions=True)
if model is not None:
    tokens, k = ci.make_repeated_tokens(tok, n_unique=20, seed=0)
    grid, base = ci.per_head_ablation(model, tokens, k)   # 逐头消融的 loss 增量
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(grid, cmap='Reds', aspect='auto')
    ax.set_xlabel('头'); ax.set_ylabel('层'); ax.set_title('逐头消融对 induction 的损害\\n(越红=该头对 induction 越关键)')
    plt.colorbar(im, label='induction loss 增量'); plt.tight_layout(); plt.show()
    top = np.dstack(np.unravel_index(np.argsort(-grid, axis=None)[:5], grid.shape))[0]
    print(f'完整 induction loss = {base:.3f}')
    print('对 induction 最关键的 top5 头 (层,头):')
    for L,H in top:
        print(f'  层{L} 头{H}: 消融损害 {grid[L,H]:+.3f}')
    print('→ 多个头都亮 → induction 是一组头协作 (冗余 circuit), 不是单个头 (L3)。')
else:
    print('无 gpt2')"""),
    md("## 2. 整组 vs 单个: 冗余的证据"),
    code("""if model is not None:
    # 单头损害 vs 多头总损害对比
    single = grid.max()
    topk_sum = np.sort(grid, axis=None)[-3:].sum()
    print(f'单个最强头损害:     {single:+.3f}')
    print(f'top3 头损害之和:    {topk_sum:+.3f}')
    print(f'''
解读 (L3/L4):
  - 单头消融损害小 (冗余: 别的头能顶上)
  - 多头合起来贡献大 (induction 是一组头的协作)
  → 这就是为什么 circuit 分析要看「整组」而非单点 (单点消融会低估)
  → 归因 patching (L4) 用梯度一次性扫全部头, 看到整个 circuit 的分布''')"""),
    md("""## 3. 反思 (12.5 收口)

你画了 induction circuit 的组件分布 (逐头贡献)。带走:
- **逐头贡献热图**: 每个头对行为的因果贡献; 多个头亮 = 冗余 circuit。
- **冗余**: 单头消融损害小 (有备份), 整组贡献大 → circuit 分析要看整组 (L3)。
- **归因 patching** (L4): 梯度一次性估全部组件贡献, 规模化 (逐头消融的加速版)。

> **M12.5 收口**: head=QK+OV; induction head=ICL机制(真gpt2); circuit=组件+信息流; 冗余要看整组; 归因 patching 规模化。
> **交棒 M12.6「cot-faithfulness-oversight」**: interp 用到推理模型 — CoT 是真「内心独白」吗? 忠实性/监控/欺骗检测 (对齐安全前沿)。下一专题 `cot-faithfulness-oversight`。"""),
]
nbformat.write(n2, HERE / "N2-head-attribution.ipynb")
print("written N2")
