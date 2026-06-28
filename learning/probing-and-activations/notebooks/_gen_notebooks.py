"""生成 12.2 notebooks (N1 线性探针玩具+gpt2 / N2 logit lens 玩具+gpt2). 跑后 nbconvert --execute。"""
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
import probing as pr
import tiny_transformer as tt
import numpy as np, torch"""

# ── N1 linear-probes ──
n1 = new_notebook()
n1.cells = [
    md("""# N1 · 线性探针: 读出被线性编码的概念

> 配套 12.2-L2 · 训线性探针从激活读概念。
> ① 玩具 transformer: 读「当前值」(各层) ② **真实 gpt2**: 读「这个 token 是不是数字」。
> 高准确率 = 概念被线性编码 (在一个方向上)。"""),
    code(PATHS + """
import realmodels as rm
print('真实模型:', rm.available())"""),
    md("## 1. 玩具 transformer: 从各层 residual 读「当前值」"),
    code(MPL + """
torch.manual_seed(0)
Xi, Yi = tt.make_data(2000, seed=0)
model = tt.build_model(); tt.train(model, Xi, Yi, epochs=800)
Xtest, _ = tt.make_data(600, seed=5)
layers = ['resid_pre','resid_post_0','resid_post_1']; accs=[]
for key in layers:
    acts, labels = pr.tiny_layer_activations(model, Xtest, layer_key=key)
    tr, te = pr.linear_probe(acts, labels, n_classes=tt.V)
    accs.append(te); print(f'  {key:14}: 探针测试准确率 {te:.2f}')
fig,ax=plt.subplots(figsize=(6,3.6)); ax.bar(layers, accs, color='C0'); ax.set_ylim(0,1.05)
ax.set_ylabel('探针准确率'); ax.set_title('玩具: 各层 residual 线性编码了「当前值」'); plt.tight_layout(); plt.show()"""),
    md("## 2. 真实 gpt2: 探针读「这个 token 是不是数字」"),
    code("""tok, gpt2 = rm.gpt2()
if gpt2 is not None:
    SENTS = [
        "The year 1999 had 3 cats and 42 happy dogs.",
        "She bought 7 apples and 18 oranges yesterday.",
        "In 2024 the team scored 5 goals in 90 minutes.",
        "He read 12 books and wrote 4 long essays.",
        "The room held 256 chairs and 30 round tables.",
        "Water freezes and the bus arrives at 8 sharp.",
    ]
    LAYER = 6
    X, y = [], []
    for s in SENTS:
        ids = tok(s, return_tensors='pt')
        with torch.no_grad():
            out = gpt2(**ids, output_hidden_states=True)
        hs = out.hidden_states[LAYER][0]   # (seq, 768) 第LAYER层 residual
        for j, t in enumerate(ids.input_ids[0]):
            txt = tok.decode([t]).strip()
            X.append(hs[j].numpy()); y.append(1 if any(c.isdigit() for c in txt) else 0)
    X = np.array(X); y = np.array(y)
    print(f'gpt2 第{LAYER}层激活 {X.shape}, 数字token {y.sum()}/{len(y)}')
    tr, te = pr.linear_probe(X, y, n_classes=2, epochs=400)
    base = max(y.mean(), 1-y.mean())
    print(f'探针读「是否数字」: 测试准确率 {te:.2f} (基线={base:.2f})')
    print('→ 探针能从 gpt2 激活线性读出「是不是数字」→ 这个概念被线性编码 (一个方向)。')
else:
    print('无 gpt2, 跳过')"""),
    md("""## 3. 反思
你训了线性探针, 从玩具 + 真实 gpt2 激活读出概念。带走:
- **线性探针**: 线性分类器从激活读概念; 高准确率 = 概念被**线性编码** (一个方向)。
- 真实 gpt2 把「是不是数字」这种概念线性编码了 (探针读得出) —— feature=方向是真的。
- 探针**必须线性** (否则探针自己算概念, 不可信)。
> ⚠ 但: 探针读出 ≠ 模型在用 (相关非因果, L4)! 验证因果要靠 M12.3 干预。
下一步 N2: logit lens 看预测逐层成形。"""),
]
nbformat.write(n1, HERE / "N1-linear-probes.ipynb")
print("written N1")

# ── N2 logit-lens ──
n2 = new_notebook()
n2.cells = [
    md("""# N2 · logit lens: 每层在「想」什么

> 配套 12.2-L3 · 把中间层 residual 过最终 unembed, 看预测如何逐层成形 (零训练)。
> ① 玩具: +1 计算在哪层成形 ② **真实 gpt2**: 看「Paris」在哪层冒出来。"""),
    code(PATHS + "\nimport realmodels as rm\nprint('真实模型:', rm.available())"),
    md("## 1. 玩具 transformer: +1 计算在哪层成形"),
    code("""torch.manual_seed(0)
Xi, Yi = tt.make_data(2000, seed=0)
model = tt.build_model(); tt.train(model, Xi, Yi, epochs=800)
sample = tt.make_data(1, seed=7)[0]
true_next = int((sample[0,-1]+1) % tt.V)
lens = pr.logit_lens_tiny(model, sample)
print(f'序列 {sample[0]}, 正确下一个 = {true_next}')
for key, pred in lens.items():
    mark = '✓ 正确' if int(pred[0])==true_next else '✗ (还没算出)'
    print(f'  {key:14}: top 预测 = {int(pred[0])}  {mark}')
print('→ resid_pre(刚embed)还没算+1; resid_post_0(第0层后)就对了 → +1 计算在第0层完成。')"""),
    md("## 2. 真实 gpt2: 看「Paris」在哪层冒出来"),
    code(MPL + """
tok, gpt2 = rm.gpt2()
if gpt2 is not None:
    ctx = "The Eiffel Tower is in the city of"
    ids = tok(ctx, return_tensors='pt')
    with torch.no_grad():
        out = gpt2(**ids, output_hidden_states=True)
    hs = out.hidden_states   # tuple(L+1) 每个 (1,seq,768)
    target = tok(' Paris').input_ids[0]   # 'Paris' 的 token id
    layer_top = []; paris_rank = []
    for L in range(len(hs)):
        resid = hs[L][0, -1]                       # 最后位置 residual
        with torch.no_grad():
            logits = gpt2.lm_head(gpt2.transformer.ln_f(resid))   # logit lens
        top = tok.decode([int(logits.argmax())]).strip()
        layer_top.append(top)
        rank = int((logits > logits[target]).sum())   # Paris 的排名 (0=第一)
        paris_rank.append(rank)
    print(f'上下文: {ctx!r}  目标: Paris')
    for L,(t,r) in enumerate(zip(layer_top, paris_rank)):
        bar = '█'*max(0, 12-min(12, r//50))
        print(f'  层{L:2}: top={t!r:12} Paris排名={r:5} {bar}')
    plt.figure(figsize=(7,3.6))
    plt.plot(range(len(paris_rank)), paris_rank, 'o-'); plt.gca().invert_yaxis()
    plt.xlabel('层'); plt.ylabel('Paris 的排名 (越小越靠前)'); plt.title('logit lens: 「Paris」逐层浮现为 top 预测')
    plt.tight_layout(); plt.show()
    print('→ 早层 top 是无关词; 越深 Paris 排名越靠前 → 事实预测在中后层成形 (零训练偷看)。')
else:
    print('无 gpt2, 跳过')"""),
    md("""## 3. 反思 (12.2 收口)

你用 logit lens 看了预测的成形过程。带走:
- **logit lens**: 中间 residual 过最终 unembed, 零训练看「每层想预测什么」。
- **预测成形**: 玩具的 +1 在第0层成形; gpt2 的「Paris」逐层浮现 (中后层定下来)。
- vs 探针: logit lens 读预测进度 (零训练), 探针读任意概念 (要训); 都是**相关** (非因果)。

> **M12.2 收口**: residual=黑板; 探针读方向(概念是否线性编码); logit lens 看预测成形; 但都只证相关, 必须走向干预。
> **交棒 M12.3「causal-interventions」**: mech interp 最核心工具 — activation patching, 因果定位「哪个组件真负责一个行为」。下一专题 `causal-interventions`。"""),
]
nbformat.write(n2, HERE / "N2-logit-lens.ipynb")
print("written N2")
