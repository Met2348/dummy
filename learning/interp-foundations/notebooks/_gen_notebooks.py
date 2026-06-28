"""生成 12.1 notebooks (N1 真实gpt2多义神经元 / N2 tiny model解剖基座). 跑后 nbconvert --execute。"""
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
sys.path.insert(0, str(Path.cwd().parents[1] / "_shared"))   # 真实模型 helper
import numpy as np, torch"""

# ── N1 real gpt2 polysemantic neurons ──
n1 = new_notebook()
n1.cells = [
    md("""# N1 · 真实 gpt2 的多义神经元 (superposition)

> 配套 12.1-L2 · **小而真**: 在真实 gpt2 (124M) 上看「多义神经元」—— 一个神经元对**多个不相关概念**都激活。
> 这是 superposition 的现象, 也是 mech interp 的头号难题 (不能一个神经元一个神经元地读)。"""),
    code(PATHS + """
import realmodels as rm
print('真实模型可用性:', rm.available())"""),
    md("> 若 gpt2=False 表示无本地缓存, 本 notebook 会跳过真实部分 (不影响课程)。"),
    md("## 1. 在一小批多样句子上, 记录 gpt2 某层 MLP 神经元的激活"),
    code("""tok, model = rm.gpt2()
SENTENCES = [
    "The cat sat on the warm mat.",
    "She invested money in the stock market.",
    "Water boils at one hundred degrees.",
    "The president signed the new law.",
    "He scored a goal in the final match.",
    "The recipe needs two cups of flour.",
    "Astronomers discovered a distant galaxy.",
    "The lawyer argued the case in court.",
    "Children played happily in the park.",
    "The engine roared as the car sped up.",
]
LAYER = 6
neuron_acts = []   # 每条: (token_str, neuron_vector)
if model is not None:
    cap = {}
    h = model.transformer.h[LAYER].mlp.act.register_forward_hook(lambda m,i,o: cap.__setitem__('a', o.detach()))
    for s in SENTENCES:
        ids = tok(s, return_tensors='pt')
        with torch.no_grad():
            model(**ids)
        a = cap['a'][0]   # (seq, 3072) 该层 MLP 隐单元(神经元)激活
        for j, t in enumerate(ids.input_ids[0]):
            neuron_acts.append((tok.decode([t]).strip() or '␣', a[j].numpy()))
    h.remove()
    print(f'记录了 {len(neuron_acts)} 个 (token, 神经元激活) 对; 每个神经元向量维度 {neuron_acts[0][1].shape[0]}')
else:
    print('无 gpt2, 跳过')"""),
    md("## 2. 挑一个神经元, 看它对哪些 token 最激活 (多义?)"),
    code("""if neuron_acts:
    import numpy as np
    acts = np.stack([v for _,v in neuron_acts])      # (N_tokens, 3072)
    toks = [t for t,_ in neuron_acts]
    # 找一个"高方差且 top token 多样"的神经元
    def top_tokens(nid, k=6):
        order = np.argsort(-acts[:, nid])[:k]
        return [(toks[i], float(acts[i, nid])) for i in order]
    # 扫几个神经元, 展示多义
    import numpy as np
    cand = np.argsort(-acts.std(0))[:40]             # 高方差神经元更可能"有内容"
    for nid in cand[[0, 5, 12]]:
        print(f'神经元 #{nid} 的 top 激活 token:')
        for t,v in top_tokens(int(nid)):
            print(f'    {t!r:14} 激活 {v:+.2f}')
        print()
    print('→ 同一个神经元的 top token 往往横跨不相关的词/概念 = 多义神经元 (superposition)。')
else:
    print('跳过')"""),
    md("## 3. 可视化: 一个神经元在不同 token 上的激活分布"),
    code(MPL + """
if neuron_acts:
    nid = int(cand[0])
    vals = acts[:, nid]
    order = np.argsort(-vals)[:14]
    fig, ax = plt.subplots(figsize=(9,4))
    ax.bar(range(len(order)), vals[order], color='C0')
    ax.set_xticks(range(len(order))); ax.set_xticklabels([toks[i] for i in order], rotation=45, ha='right')
    ax.set_title(f'gpt2 第{LAYER}层 神经元 #{nid} 对各 token 的激活 (top14)'); ax.set_ylabel('激活')
    plt.tight_layout(); plt.show()
    print('→ 这一个神经元对一堆不相关 token 都激活 → 它不是"一个概念一个神经元"。')"""),
    md("""## 4. 反思
你在**真实 gpt2** 上看到了多义神经元 (superposition 的现象)。带走:
- **superposition**: 网络把多于神经元数的 feature 叠加压缩 → 单个神经元**多义** (对多个不相关概念激活)。
- 后果: **不能一个神经元一个神经元地读** —— 这是 mech interp 的头号难题。
- 解法方向: 读「方向」而非神经元 (线性探针 M12.2)、解叠加 (SAE M12.4)。
下一步 N2: 在受控玩具 transformer 上看激活如何编码概念 (为 probing 准备)。"""),
]
nbformat.write(n1, HERE / "N1-polysemantic-neurons.ipynb")
print("written N1")

# ── N2 tiny model dissection base ──
n2 = new_notebook()
n2.cells = [
    md("""# N2 · 受控玩具 transformer: 解剖基座

> 配套 12.1-L3 · 训一个**结构已知**的 tiny transformer (任务: increment-mod-V, 下一个=当前+1)。
> 它的中间激活全可读取 (run_with_cache), 是后续 probing/patching/circuits 的受控解剖对象。
> 受控玩具 = 有 ground truth, 教方法; 真 gpt2 (N1) = 验证可迁移。"""),
    code(PATHS + "\nimport tiny_transformer as tt\nprint('tiny transformer: 词表', tt.V, '层', tt.N_LAYERS, '头', tt.N_HEADS)"),
    md("## 1. 训练玩具 transformer 学 increment-mod-V"),
    code("""torch.manual_seed(0)
Xi, Yi = tt.make_data(2000, seed=0)
print('任务样例 (输入→目标, 下一个=当前+1 mod V):')
print('  输入 ', Xi[0]); print('  目标 ', Yi[0])
model = tt.build_model()
losses = tt.train(model, Xi, Yi, epochs=800)
acc = tt.accuracy(model, *tt.make_data(500, seed=9))
print(f'训练 loss {losses[0]:.3f} → {losses[-1]:.3f}, 测试准确率 {acc:.2f}')"""),
    md("## 2. 读取中间激活 (run_with_cache) —— 解剖的入口"),
    code("""logits, cache = model.run_with_cache(torch.tensor(Xi[:1]))
print('可读取的激活 (residual stream / attention / MLP, 每层):')
for k in cache: print(f'    {k:18} {tuple(cache[k].shape)}')
print('\\n→ 每个组件从 residual stream 读写信息, 全部可读取 (L3 的逆向工程入口)。')"""),
    md("## 3. residual stream 编码了「当前值」吗 (probing 预览)"),
    code(MPL + """
# 收集很多序列在 最后一层 residual 的激活, 看是否按"当前 token 值"聚类
Xb, Yb = tt.make_data(300, seed=3)
_, cb = model.run_with_cache(torch.tensor(Xb))
resid = cb['resid_post_1'][:, -1, :].numpy()    # 最后位置的 residual (B, d_model)
cur_val = Xb[:, -1]                               # 最后位置的"当前值"
# PCA 到 2D 上色
resid_c = resid - resid.mean(0)
U,S,Vt = np.linalg.svd(resid_c, full_matrices=False)
proj = resid_c @ Vt[:2].T
fig, ax = plt.subplots(figsize=(6,5))
sc = ax.scatter(proj[:,0], proj[:,1], c=cur_val, cmap='hsv', s=18)
plt.colorbar(sc, label='当前 token 值'); ax.set_title('residual stream (PCA) 按"当前值"聚类\\n→ 概念被线性编码 (probing 能读出, M12.2)')
plt.tight_layout(); plt.show()
print('→ residual 里"当前值"这个概念被结构化编码 (同色聚集) → M12.2 线性探针能读出它。')"""),
    md("""## 4. 反思 (12.1 收口)

你准备好了 mech interp 的解剖基座。带走:
- **受控玩具 transformer**: 结构已知 (increment-mod-V) + 激活全可读 → 教 interp 方法的干净对象。
- **residual stream**: 信息主干, 每组件读写; 概念 (当前值) 被结构化编码 (probing 预览)。
- **两条腿**: 受控玩具 (有 ground truth, 教方法) + 真 gpt2 (N1, 验证可迁移)。

> **M12.1 收口**: mech interp=逆向工程网络; feature/circuit/superposition; 多义神经元是难题; residual stream 是数据总线。
> **交棒 M12.2「probing-and-activations」**: 第一件工具 — 线性探针 + logit lens, 从激活读出概念 (但探针是相关, 引出 12.3 干预)。下一专题 `probing-and-activations`。"""),
]
nbformat.write(n2, HERE / "N2-toy-model-dissection.ipynb")
print("written N2")
