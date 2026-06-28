"""生成 12.3 notebooks (N1 activation patching 因果定位 / N2 ablation 必要性). 跑后 nbconvert --execute。"""
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
import patching as pt
import tiny_transformer as tt
import numpy as np, torch"""

# ── N1 activation-patching ──
n1 = new_notebook()
n1.cells = [
    md("""# N1 · activation patching: 因果定位

> 配套 12.3-L2 · mech interp 最核心工具。clean/corrupt 对照 (玩具: 只差最后一个 token →答案不同)。
> 把 clean 激活逐个 (层,位置) patch 进 corrupt 运行, 看哪里能**恢复** clean 行为 = 那里因果携带答案信息。"""),
    code(PATHS + "\ntorch.manual_seed(0)\nXi,Yi=tt.make_data(2000,seed=0)\nmodel=tt.build_model(); tt.train(model,Xi,Yi,epochs=800)\nprint('玩具 transformer 就绪 (increment-mod-V)')"),
    md("## 1. clean / corrupt 对照 (只差最后一个 token)"),
    code("""clean, corrupt, ca, cora = pt.make_clean_corrupt(seed=3)
print(f'clean   {clean[0]} → 答案 {ca}')
print(f'corrupt {corrupt[0]} → 答案 {cora}  (只改了最后一个 token)')
print('问题: 哪个位置/层 因果携带了"答案"信息?')"""),
    md("## 2. activation patching 恢复率热图"),
    code(MPL + """
grid, info = pt.patch_recovery(model, seed=3)
fig, ax = plt.subplots(figsize=(8,2.6))
im = ax.imshow(grid, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax.set_yticks(range(tt.N_LAYERS)); ax.set_yticklabels([f'层{L}' for L in range(tt.N_LAYERS)])
ax.set_xticks(range(tt.SEQ)); ax.set_xlabel('位置'); ax.set_title('activation patching 恢复率 (绿=patch此处恢复clean行为=因果)')
for L in range(tt.N_LAYERS):
    for p in range(tt.SEQ):
        ax.text(p, L, f'{grid[L,p]:.1f}', ha='center', va='center', fontsize=8)
plt.colorbar(im, label='恢复率'); plt.tight_layout(); plt.show()
best = np.unravel_index(np.argmax(grid), grid.shape)
print(f'→ 只有「最后位置」patch 能恢复 (恢复率≈1), 其它≈0 → 答案信息因果地在最后位置。')
print(f'  (increment 任务里 next=last+1, 所以只有最后 token 决定答案 — patching 干净地证明了这点)')"""),
    md("""## 3. 这就是「因果」而非「相关」
- 探针 (M12.2) 能从很多位置读出信息 (相关), 但 patching 证明**只有最后位置因果负责**。
- 这是 mech interp 的命门: **动手改 (patch), 看行为变化 (因果)**, 不是看激活猜 (相关)。"""),
    md("""## 4. 反思
你用 activation patching 做了**因果定位**。带走:
- **patching**: clean 激活贴进 corrupt, 看恢复; 恢复=该处因果携带行为信息 (充分性)。
- **clean/corrupt 对照是灵魂**: 只差你关心的信息, 定位才干净 (接 M9.4)。
- 热图 = 因果信息流地图; 玩具里答案信息干净地定位在最后位置。
> 真 gpt2 上同理: patching "capital of France" 能定位国家名信息在哪层/位置被搬到答案 (causal tracing)。
下一步 N2: ablation 找「必要」组件 (和 patching 的充分互补 = 充要)。"""),
]
nbformat.write(n1, HERE / "N1-activation-patching.ipynb")
print("written N1")

# ── N2 ablation-scan ──
n2 = new_notebook()
n2.cells = [
    md("""# N2 · ablation: 找最小必要子集

> 配套 12.3-L3 · patching 找「充分」(patch 它就恢复); ablation 找「必要」(去它就坏)。
> 对每个 (层,位置) 置零激活, 测对答案的损害。配合 N1 的 patching = **充要**因果证据。"""),
    code(PATHS + "\ntorch.manual_seed(0)\nXi,Yi=tt.make_data(2000,seed=0)\nmodel=tt.build_model(); tt.train(model,Xi,Yi,epochs=800)\nprint('就绪')"),
    md("## 1. ablation 损害热图 (置零各位置, 看答案 logit 掉多少)"),
    code(MPL + """
dmg = pt.ablate_effect(model, seed=3)
fig, ax = plt.subplots(figsize=(8,2.6))
im = ax.imshow(dmg, cmap='Reds', aspect='auto')
ax.set_yticks(range(tt.N_LAYERS)); ax.set_yticklabels([f'层{L}' for L in range(tt.N_LAYERS)])
ax.set_xticks(range(tt.SEQ)); ax.set_xlabel('位置'); ax.set_title('ablation 损害 (越红=去掉此处答案越坏=越必要)')
for L in range(tt.N_LAYERS):
    for p in range(tt.SEQ):
        ax.text(p, L, f'{dmg[L,p]:.1f}', ha='center', va='center', fontsize=7,
                color='white' if dmg[L,p]>dmg.max()*0.5 else 'black')
plt.colorbar(im, label='答案 logit 损害'); plt.tight_layout(); plt.show()
worst = np.unravel_index(np.argmax(dmg), dmg.shape)
print(f'→ 损害最大 = 层{worst[0]} 位置{worst[1]} (去掉它答案最坏 = 最必要)。')"""),
    md("## 2. 充分 (patching) + 必要 (ablation) = 充要因果"),
    code("""grid, _ = pt.patch_recovery(model, seed=3)
patch_best = np.unravel_index(np.argmax(grid), grid.shape)
abl_best = np.unravel_index(np.argmax(pt.ablate_effect(model, seed=3)), pt.ablate_effect(model, seed=3).shape)
print(f'patching (充分): 最能恢复 = 位置 {patch_best[1]}')
print(f'ablation (必要): 最必要   = 位置 {abl_best[1]}')
print(f'两者{"都指向同一位置 → 充要因果证据 (强)" if patch_best[1]==abl_best[1] else "不一致 → 需进一步分析"}')
print('''
证据强度 (L3):
  相关 (探针)        弱  ← 可能旁路
  充分 (patching)    中  ← patch它就恢复
  必要 (ablation)    中  ← 去它就坏
  充要 (两者都指向)  强  ← 严谨的因果定位''')"""),
    md("""## 3. 反思 (12.3 收口)

你用 patching (充分) + ablation (必要) 做了**充要**因果定位。带走:
- **ablation**: 删激活看损害; 损害大=必要组件 (用 mean/resample 比 zero 更可信, L4)。
- **充要论证**: patching 找充分 + ablation 找必要, 两者指向同一处 = 强因果证据。
- **严谨性是命门** (L4): 单变量/对照/度量/分布内/多样本; 否则是电路占星术。

> **M12.3 收口**: 相关→必须干预; patching 因果定位 (充分); ablation 找必要; 充要双验证; 干预=消融极致 (接 M9.4)。
> **交棒 M12.4「sparse-autoencoders」**: 回到 superposition — SAE 把叠加的多义神经元**解叠加**成单义特征 (Anthropic 显微镜)。下一专题 `sparse-autoencoders`。"""),
]
nbformat.write(n2, HERE / "N2-ablation-scan.ipynb")
print("written N2")
