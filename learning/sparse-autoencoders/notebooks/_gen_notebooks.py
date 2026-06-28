"""生成 12.4 notebooks (N1 训mini-SAE解叠加 / N2 特征单义性+真实gpt2). 跑后 nbconvert --execute。"""
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
import sae as S
import tiny_transformer as tt
import numpy as np, torch"""

# ── N1 train-mini-sae ──
n1 = new_notebook()
n1.cells = [
    md("""# N1 · 训一个 mini-SAE 解叠加

> 配套 12.4-L1/L2 · 在玩具 transformer 激活上训稀疏自编码器 (过完备+稀疏), 把叠加的激活分解成稀疏特征。
> 看 SAE 特征比**原始神经元更单义** (纯度对比) —— 解叠加的硬证据。"""),
    code(PATHS + "\ntorch.manual_seed(0)\nXi,Yi=tt.make_data(2000,seed=0); model=tt.build_model(); tt.train(model,Xi,Yi,epochs=800)\nprint('玩具 transformer 就绪')"),
    md("## 1. 收集激活 (跑模型, 取某层 residual) + 训 SAE"),
    code(MPL + """
acts, labels = S.tiny_mlp_activations(model, tt.make_data(1500, seed=2)[0])
print(f'激活 {acts.shape} (d_model={acts.shape[1]}); 概念类别 {tt.V} 个; SAE 过完备到 {tt.V*3} 特征')
sae = S.build_sae(d_in=acts.shape[1], n_features=tt.V*3)
losses, sparsity = S.train_sae(sae, acts, epochs=600, l1=1e-2)
fig, ax = plt.subplots(1,2,figsize=(11,3.6))
ax[0].plot(losses); ax[0].set_title('重建损失'); ax[0].set_xlabel('epoch')
ax[1].plot(sparsity); ax[1].set_title('稀疏度 (平均激活特征比例)'); ax[1].set_xlabel('epoch')
plt.tight_layout(); plt.show()
print(f'重建 {losses[0]:.3f}→{losses[-1]:.3f}; 末期稀疏度 {sparsity[-1]:.2f} (每次只少数特征激活)')"""),
    md("## 2. SAE 特征 vs 原始神经元: 谁更单义"),
    code(MPL + """
codes = S.feature_codes(sae, acts)
active, sae_purity = S.monosemanticity(codes, labels)
_, raw_purity = S.monosemanticity(acts - acts.min(), labels)
fig, ax = plt.subplots(figsize=(6,3.8))
ax.bar(['原始神经元','SAE 特征'], [raw_purity, sae_purity], color=['C3','C0'])
for i,v in enumerate([raw_purity, sae_purity]): ax.text(i, v+0.01, f'{v:.2f}', ha='center')
ax.set_ylabel('单义纯度 (高=一个特征对应一个概念)'); ax.set_ylim(0,1)
ax.set_title(f'SAE 解叠加: 特征纯度 {sae_purity:.2f} >> 原始神经元 {raw_purity:.2f}')
plt.tight_layout(); plt.show()
print(f'活跃特征 {len(active)} 个; SAE 纯度 {sae_purity:.2f} vs 原始 {raw_purity:.2f} '
      f'(SAE 单义 {sae_purity/max(raw_purity,1e-6):.1f}×)')
print('→ SAE 把叠加的多义激活, 解成更单义的特征 (过完备+稀疏的功劳)。')"""),
    md("""## 3. 反思
你训了一个 mini-SAE 并验证它解叠加。带走:
- **SAE = 过完备 + 稀疏字典**: encode(ReLU,宽) → decode(重建); loss = 重建 + L1 稀疏。
- **解叠加硬证据**: SAE 特征纯度 远高于原始神经元 (定量, 非讲故事)。
- λ (l1) 是旋钮: 稀疏 vs 重建权衡 (你看到的稀疏度)。
> ⚠ 但 (L4): 纯度高 ≠ 特征真实/模型在用; 要因果验证 (干预特征, M12.3)。
下一步 N2: 检视特征是否单义 (最大激活样本) + 真实 gpt2 SAE。"""),
]
nbformat.write(n1, HERE / "N1-train-mini-sae.ipynb")
print("written N1")

# ── N2 feature-monosemanticity ──
n2 = new_notebook()
n2.cells = [
    md("""# N2 · 特征单义性 + 真实 gpt2 SAE

> 配套 12.4-L3 · ① 玩具: 每个 SAE 特征对应哪个「当前值」(特征×值热图, 看单义)。
> ② **真实 gpt2**: 在 gpt2 激活上训 mini-SAE, 找一个特征的最大激活 token (比原始多义神经元更连贯)。"""),
    code(PATHS + "\nimport realmodels as rm\nprint('真实模型:', rm.available())"),
    md("## 1. 玩具: 特征 × 当前值 热图 (单义 = 每个特征专一对应一个值)"),
    code(MPL + """
torch.manual_seed(0)
Xi,Yi=tt.make_data(2000,seed=0); model=tt.build_model(); tt.train(model,Xi,Yi,epochs=800)
acts, labels = S.tiny_mlp_activations(model, tt.make_data(1500, seed=2)[0])
sae = S.build_sae(acts.shape[1], tt.V*3); S.train_sae(sae, acts, epochs=600, l1=1e-2)
codes = S.feature_codes(sae, acts)
active, _ = S.monosemanticity(codes, labels)
# 每个活跃特征在各 value 上的平均激活
M = np.zeros((len(active), tt.V))
for i, fi in enumerate(active):
    for v in range(tt.V):
        M[i, v] = codes[labels==v, fi].mean()
# 按峰值 value 排序特征, 让"对角"显现
order = np.argsort(M.argmax(1))
fig, ax = plt.subplots(figsize=(7,5))
im = ax.imshow(M[order], aspect='auto', cmap='viridis')
ax.set_xlabel('当前值 (概念)'); ax.set_ylabel('SAE 特征 (按峰值排序)'); ax.set_xticks(range(tt.V))
ax.set_title('特征×值热图: 每个特征专一对应一个值 (亮点=单义)')
plt.colorbar(im); plt.tight_layout(); plt.show()
print('→ 每行(特征)只在某一列(值)亮 → 特征单义 (一个特征=一个概念)。这就是解叠加的样子。')"""),
    md("## 2. 真实 gpt2: 在激活上训 mini-SAE, 看一个特征的最大激活 token"),
    code("""tok, gpt2 = rm.gpt2()
if gpt2 is not None:
    CORPUS = [
        "The cat sat on the warm mat near the door.",
        "She invested money in the stock market last year.",
        "Water boils at one hundred degrees in the pot.",
        "The president signed the new law on Monday.",
        "He scored a goal in the final football match.",
        "The recipe needs two cups of white flour.",
        "Astronomers discovered a distant bright galaxy.",
        "The lawyer argued the difficult case in court.",
        "Children played happily in the green park.",
        "The loud engine roared as the car sped up fast.",
        "Doctors treated the patient in the busy hospital.",
        "The river flowed gently past the old stone bridge.",
    ]
    LAYER = 6
    X, meta = [], []
    h = gpt2.transformer.h[LAYER].mlp.act.register_forward_hook(lambda m,i,o: cap.__setitem__('a', o.detach()))
    cap = {}
    for s in CORPUS:
        ids = tok(s, return_tensors='pt')
        with torch.no_grad(): gpt2(**ids)
        a = cap['a'][0]   # (seq, 3072)
        for j,t in enumerate(ids.input_ids[0]):
            X.append(a[j].numpy()); meta.append(tok.decode([t]).strip() or '␣')
    h.remove()
    X = np.array(X, dtype=np.float32)
    print(f'gpt2 第{LAYER}层 MLP 激活 {X.shape}; 训 mini-SAE...')
    gsae = S.build_sae(X.shape[1], 512, seed=0)
    S.train_sae(gsae, X, epochs=400, l1=2e-3)
    gcodes = S.feature_codes(gsae, X)
    # 找几个"活跃且集中"的特征, 看最大激活 token
    act_count = (gcodes > 1e-3).sum(0)
    cand = np.where((act_count > 2) & (act_count < len(X)*0.3))[0]
    shown = 0
    for fi in cand[np.argsort(-gcodes[:, cand].max(0))]:
        top = np.argsort(-gcodes[:, fi])[:5]
        print(f'  SAE 特征 #{fi}: 最大激活 token = {[meta[i] for i in top]}')
        shown += 1
        if shown >= 4: break
    print('→ SAE 特征的最大激活 token 比 12.1 的原始多义神经元更"成主题" (解叠加的真实版)。')
else:
    print('无 gpt2, 跳过')"""),
    md("""## 3. 反思 (12.4 收口)

你检视了 SAE 特征的单义性 (玩具 + 真实 gpt2)。带走:
- **最大激活样本**: 给特征贴人话标签; 玩具特征专一对应一个值, gpt2 特征成主题。
- **解叠加**: SAE 把叠加的多义激活, 变成更单义、可读的特征字典 (Anthropic 显微镜)。
- ⚠ **批判** (L4): 特征单义 ≠ 真实/模型在用; 评估 SAE 有争议; 要因果验证 (干预)。

> **M12.4 收口**: superposition→SAE 解叠加 (过完备+稀疏); 最大激活样本贴标签; 金门大桥特征; 但评估/真实性/因果有争议。
> **交棒 M12.5「circuits-attention」**: 把单义特征**连成算法** — attention head 算什么 (QK/OV)、induction heads (in-context learning 机制)、归因。下一专题 `circuits-attention`。"""),
]
nbformat.write(n2, HERE / "N2-feature-monosemanticity.ipynb")
print("written N2")
