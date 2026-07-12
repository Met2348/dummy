# 03 · DPO 家族对齐算法横向对比 Capstone 深挖

> 素材来源：`learning/dpo-family/src/capstone_dpo_comparison.py` + `rainbowpo.py` + `dpo_minimal.py`/`ipo_minimal.py`/`kto_minimal.py`/`orpo_minimal.py`/`simpo_minimal.py`/`cpo_minimal.py`/`dpop_minimal.py` + `src/tests/test_dpo_loss_equivalence.py` + `test_six_methods_consistency.py` + `lectures/01-dpo.md`/`04-orpo.md`/`05-simpo.md`/`12-rainbowpo.md` + `notebooks/N1-po-variants-comparison.ipynb` + `papers/README.md`。
> 本文档是脚手架，不是台词稿——练法见 [`00-how-to-defend.md`](00-how-to-defend.md)。

---

## 1. 背景与目标

DPO（Direct Preference Optimization）之后，"去 RM/去 PPO"的偏好优化方法在 2023-2024 年迅速分化出一族变体：IPO（squared loss 防 over-confidence）、KTO（单边偏好，无需 pair）、ORPO（无 ref model，SFT 锚住）、SimPO（length-normalized，无 ref）、CPO（对比式 + SFT）、DPOP（hinge 修 chosen 概率下降）。这些方法各修一个具体问题，但工程上容易变成"每个方法一份独立代码，互相对不上口径"。

本专题（`dpo-family`）的目标是两层：
1. **手写实现层**：每个变体一个 `*_minimal.py`，最小可跑的 loss 实现 + 数值 smoke test。
2. **统一框架层**（`rainbowpo.py`）：证明 DPO/IPO/ORPO/SimPO/CPO/DPOP 可以用同一个 `unified_po_loss()` 函数 + 一个 4 维超参 `POConfig`（`use_ref` / `length_norm` / `loss_type` / `add_sft`）表达，而不是 6 份互相独立的代码。

Capstone（`capstone_dpo_comparison.py`）在此基础上做**同基座、同数据形状、同训练步数**的横向对比：跑 6 个变体各 50 步的 mock 梯度更新，观察 loss、reward margin（chosen − rejected）、chosen 概率漂移三条曲线，用来具象化"DPO 的 chosen 概率可能下降""SimPO 无长度漂移"这些理论断言。

---

## 2. 个人贡献

这是自学项目，独立完成的部分：
- 7 个变体的手写 loss 实现（`dpo_minimal.py` 到 `dpop_minimal.py`），每个都带一个可独立运行的数值 smoke test（`python xxx_minimal.py` 直接跑出示例数字）。
- `rainbowpo.py` 的统一接口：把 7 个变体归纳成 4 个正交超参维度（`use_ref`/`length_norm`/`loss_type`/`add_sft`），写出一个 `unified_po_loss()` 覆盖全部变体。
- `capstone_dpo_comparison.py`：设计 `mock_step()` 单步模拟（复用 `unified_po_loss` 反传，只对 chosen/rejected 的 actor log-prob 两个标量做梯度更新）+ `benchmark()` 跑 6 变体 × 50 步 + `print_table()` 输出对比表。
- 14 个单元测试（`test_dpo_loss_equivalence.py` 4 个 + `test_six_methods_consistency.py` 10 个），覆盖数值正确性（零 margin → log2）、边界行为（DPOP 惩罚 chosen 下降、SimPO 长度归一不受长度影响）、以及 RainbowPO 统一公式与手写 DPO 的数值一致性。
- 定位并修复了一个真实 bug（见第 4 节），commit `5f977cf`。

---

## 3. 关键技术决策与理由

### 3.1 为什么统一到 4 个超参维度，而不是 7 份独立代码

`rainbowpo.py` 把每个变体拆成：
```python
@dataclass
class POConfig:
    use_ref: bool       # 是否用 ref model
    length_norm: bool   # 是否 length normalize
    loss_type: str       # "sigmoid" | "squared" | "hinge"
    add_sft: bool        # 是否加 NLL on chosen
    beta: float = 0.1
    lambda_sft: float = 1.0
```
7 个具名变体的实际取值（`VARIANTS` 字典）：

| 变体 | use_ref | length_norm | loss_type | add_sft | β | λ_sft |
|---|---|---|---|---|---|---|
| dpo | T | F | sigmoid | F | 0.1 | 1.0 |
| ipo | T | F | squared | F | 0.1 | 1.0 |
| orpo | F | F | sigmoid | T | **1.0** | **10.0** |
| simpo | F | T | sigmoid | F | **2.5** | 1.0 |
| cpo | F | F | sigmoid | T | 0.1 | **2.0** |
| kto | T | F | sigmoid | F | 0.1 | 1.0 |
| dpop | T | F | sigmoid | F | 0.1 | 1.0 |

理由：DPO 系变体大多数只是"要不要 ref model / 要不要按长度归一 / 用什么形状的偏好损失 / 要不要额外加 SFT 锚"这四个二/三值选择的组合（lecture `12-rainbowpo.md` Slide 2：2×2×3×2=24 种组合，7 个被命名）。写成统一接口的直接工程收益：一份 loss + 一份 trainer，切换变体是改 yaml 而不是改代码（Slide 16）。

**一个诚实的技术细节（会被追问）**：`kto` 在 `VARIANTS` 里的 4 个超参和 `dpo` **完全相同**（`True, False, "sigmoid", False`，同样 β=0.1）。这不是疏漏，而是这个统一框架的真实局限：KTO 的数学本质是单边（只有 `is_desirable` 标签，没有 chosen/rejected pair），`unified_po_loss()` 的接口天然要求一对 chosen/rejected log-prob，**没法真正表达 KTO 的单边形式**——`kto_minimal.py` 里的 `kto_loss()` 用的是完全不同的函数签名（`log_p_actor, log_p_ref, is_desirable`）。这正是为什么 `capstone_dpo_comparison.py` 的横向对比**只跑 6 个变体**（`["dpo", "ipo", "orpo", "simpo", "cpo", "dpop"]`），显式排除了 `kto`——标题写"6 PO 方法"是有意为之，不是漏了第 7 个。

**更值得深挖的一点**：`dpop` 在 `VARIANTS` 里的 4 个超参**同样和 `dpo` 完全相同**（`True, False, "sigmoid", False`，β=0.1、λ_sft=1.0 均为默认值）——`POConfig` 根本没有 DPOP 需要的 hinge 惩罚项（`lambda_p` 这个字段在 `dpop_minimal.py` 里存在，但 `POConfig` 数据类里没有对应字段），`loss_type` 也仍是 `"sigmoid"` 而不是 lecture `12-rainbowpo.md` Slide 4 表格里自己写的 "0.1 **(+ hinge)**"。也就是说：**RainbowPO 统一框架里的 "dpop" 预设，实际计算出来的 loss 和 "dpo" 完全一样**，真正的 DPOP hinge 逻辑只存在于独立的 `dpop_minimal.py::dpop_loss()` 里，从未被接入 `unified_po_loss()`。这不是我的推测——`notebooks/N1-po-variants-comparison.ipynb` 实际跑出来的数字直接证实了这一点：50 步之后 `DPO` 和 `DPOP` 的最终 margin **都是 -0.001，完全相同**（见第 5 节）。lecture 表格承诺了"+hinge"，代码没有兑现。

### 3.2 ORPO 在统一公式里是简化版，不是真正的 odds-ratio

`orpo_minimal.py` 里真正的 ORPO 用 **log-odds** 替代 log-ratio：
```python
def log_odds(log_p):
    log_p = log_p.clamp(max=-1e-6)
    return log_p - torch.log1p(-log_p.exp())   # log(p/(1-p))
```
但 `rainbowpo.py` 的 `unified_po_loss()` 对 orpo 预设（`use_ref=False, loss_type="sigmoid"`）算的是**普通 log-ratio**（`log_ratio_c = log_p_c_actor`，因为 `use_ref=False`），不是 log-odds 变换。也就是说，RainbowPO 统一框架捕捉到的是 ORPO"无 ref + SFT 锚住"这个**工程性质**，但没有复刻它 log-odds 的**具体数学形式**——这是论文原意的近似，不是逐公式等价迁移。lecture `12-rainbowpo.md` Slide 13 也承认"ORPO 不是严格 DPO 等价，而是另一种凸推 preference 的方式"。

### 3.3 mock_step 的设计：为什么只对两个标量反传

`mock_step()` 复用 `unified_po_loss`，但只让 `log_p_c_a`（chosen, actor）和 `log_p_r_a`（rejected, actor）两个张量 `requires_grad_(True)` 并接受梯度更新；`log_p_c_r`/`log_p_r_r`（ref model 的 log-prob）和 `sft_l`（SFT loss）保持不变。这对应真实训练里 ref model **冻结**、只有 actor 更新的设定——用两个可学习标量模拟"actor 的 chosen/rejected log-prob 会怎么变"，而不需要真的跑一个 transformer 前向。这是刻意的简化：**验证的是 6 个变体 loss 的更新方向差异，不是真实序列级训练**（见第 5 节局限）。

### 3.4 β 取值差异是刻意设计，不是随便填的默认值

`orpo` β=1.0（+ λ_sft=10.0，SFT 主导）、`simpo` β=2.5（远高于 DPO 默认的 0.1）、`cpo` λ_sft=2.0——这些不是占位符，对应各自论文的真实超参惯例（lecture `05-simpo.md` Slide 6："β 2.5（而非 DPO 的 0.1），注：β 大是因为 length-normalized log_p 数值小"）。

---

## 4. 踩过的坑与解决

**真实记录的 bug**（commit `5f977cf`，2026-06-28，"修 dpo capstone bug"）：

`mock_step()` 每一步返回一个合并字典：
```python
return {k: v.detach() for k, v in s.items()} | {"loss": out["total"].item(), "margin": out["margin_mean"].item()}
```
这个返回值会作为**下一次调用**的 `init_state` 传进去。问题是：第二次调用时，`init_state` 里除了原本的张量键（`log_p_c_a` 等）之外，还多了上一步塞进去的 `"loss"` 和 `"margin"` 两个 **Python float**（来自 `.item()`）。而 `mock_step()` 开头无条件对 `init_state` 每个值调用 `.clone()`：
```python
s = {k: v.clone() for k, v in init_state.items()}   # 旧代码
```
float 没有 `.clone()` 方法 → 第二步就 `AttributeError`。同样的问题也出现在结尾的 `.detach()` 调用上。

**修复**：给 clone/detach 都加上 `hasattr` 判断，非张量值原样透传：
```python
s = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in init_state.items()}
...
return {k: (v.detach() if hasattr(v, "detach") else v) for k, v in s.items()} | {...}
```
这是一个典型的"字典累积额外键导致类型假设被打破"的 bug——本质原因是把"训练状态"和"本步统计结果"塞进同一个字典里传递，两者生命周期语义不同。诚实说：这个坑在代码合入时就应该被 6 变体循环的第二次迭代直接触发（不需要边界数据），说明最初写完 `mock_step` 后大概率只手动跑了一次就没继续跑 `benchmark()` 的完整 50 步循环去验证。

**（推测，非文档明确记载）** 另一个潜在但代码里未显式记录为"坑"的点：`rainbowpo.py` 的 `orpo_loss` 分支里 `log_odds()` 依赖 `log_p.clamp(max=-1e-6)` 防止 `log(0)`——这个数值稳定 trick 在 lecture `04-orpo.md` Slide 8 里被专门列出来讲，暗示这在实现时是真实踩过的数值坑（log_p 太接近 0 会导致 `log1p(-exp(log_p))` 溢出/NaN），但 capstone 和测试代码本身没有对这个边界写专门的回归测试。

---

## 5. 结果与诚实局限

### 结果（真实跑出来的数字）

`notebooks/N1-po-variants-comparison.ipynb` 实际执行 `dc.benchmark(steps=50)` 后打印的 6 变体最终 margin（chosen − rejected，训练 50 步后）：
```
DPO    : -0.001
IPO    : +0.001
ORPO   : -0.024
SIMPO  : -0.032
CPO    : -0.017
DPOP   : -0.001
```
这些数字全部接近 0（多数为负），**这是预期之中的**：`benchmark()` 用 `torch.manual_seed(0)`、`B=16, T=12`、`lr=0.05`，初始 log-prob 是 `randn(B) - 5` 的随机噪声，50 步单标量梯度下降本身就是一个非常粗糙的合成模拟，数字大小**不代表真实偏好训练的 margin 增长**，只用来对比 6 个变体之间**更新方向**的相对差异（比如 DPOP 是否真的比 DPO 更少地让 chosen 概率下降）。

测试结果：
- `test_dpo_loss_equivalence.py`：4 个测试全部通过（零 margin loss=log2≈0.6931、正 margin loss 变小、β 放大效应、负 margin loss 变大）。
- `test_six_methods_consistency.py`：10 个测试全部通过，包括 `test_rainbowpo_dpo_matches`（统一公式与手写 DPO loss 数值误差 < 1e-5）、`test_dpop_punishes_chosen_drop`（构造 chosen 概率从 0.5→0.3 的反例，验证 DPOP loss > DPO loss）、`test_simpo_length_normalize`（构造不同长度但等价 per-token log_p 的输入，验证归一化后数值一致，`atol=1e-6`）。
- 合计 **14/14 测试通过**（实测 1.66s，2026-07 本次复核运行）。

### 诚实局限

1. **DPOP 在统一框架里其实没有被真正实现**——`VARIANTS["dpop"]` 的 4 个超参和 `VARIANTS["dpo"]` 逐字段相同，`unified_po_loss()` 对两者算出来的 loss 数学上恒等。`notebooks/N1-po-variants-comparison.ipynb` 的实测输出直接证明了这一点：DPO 和 DPOP 50 步后的最终 margin **都是 -0.001**。capstone 的对比表面上"6 个变体都不同"，但其中 DPO/DPOP 这一对在统一框架路径下是重复的；真正体现"DPOP 修复 chosen 概率下降"这个设计意图的，只有测试文件里单独调用 `dpop_minimal.py::dpop_loss()`（走独立实现，不走 `unified_po_loss`）的 `test_dpop_punishes_chosen_drop`。这是本 capstone 结果里最容易被忽略、但一旦被追问就很难圆过去的一个具体缺口。
2. **"统一公式数值一致"目前只验证了 DPO 一个变体**。`test_rainbowpo_dpo_matches` 是测试文件里唯一一条"统一公式 vs 手写实现"的交叉验证测试；IPO/ORPO/SimPO/CPO/DPOP 各自只有独立的结构性断言（比如"loss > 0""loss 比 SFT 大"），**没有**类似的"unified_po_loss(cfg=VARIANTS["simpo"]) 是否等于 simpo_loss()"这类交叉一致性测试。也就是说，RainbowPO 统一框架对 DPO 之外 5 个变体的"化简是否正确"，目前主要靠人工推导 + lecture 里的公式对照，**测试覆盖不完整**。
3. **capstone 的"6 方法横向对比"是合成标量模拟，不是真实模型训练**。`mock_step` 操作的是随机初始化的标量 log-prob，不涉及真实 tokenizer、真实 transformer 前向、真实偏好数据集；lecture `01-dpo.md` Slide 15 预告的"L13 capstone 同基座同数据跑 6 个 DPO 变体"里"同基座"（Qwen2.5-0.5B）、"同数据"（Anthropic-HH）在代码里**只是模块顶部 docstring 里写的"模拟"字样**（`"""同基座 (Qwen2.5-0.5B 模拟) + 同数据 (Anthropic-HH mock) + 同 step."""`），实际并未加载真实模型或数据集。
4. **只有 DPO 有真正的端到端训练脚本**。`dpo_minimal.py` 的 `train()` 会真实调用 `AutoModelForCausalLM`/`AutoTokenizer` 加载 gpt2，并尝试 `load_dataset("Anthropic/hh-rlhf", ...)`（失败则退化为 dummy 数据）。其余 5 个变体（ipo/kto/orpo/simpo/cpo/dpop）的 `*_minimal.py` **只有 loss 函数 + 随机张量的数值 smoke test**，没有 `train()` 循环，从未在真实模型上跑过。
5. **lecture 里提到的"三轨实现"（手写/trl/yaml）只有"手写"一轨真实存在**。`01-dpo.md` Slide 17/25、`04-orpo.md` Slide 13、`05-simpo.md` Slide 10 都提到 `dpo_trl.py`/`orpo_trl.py`/`simpo_trl.py`（用 `trl.DPOTrainer`/`ORPOTrainer`/`CPOTrainer`），但 `src/` 目录下**不存在任何 `*_trl.py` 文件**——这部分只是 lecture 文本里的示例代码块，未落地成可运行脚本。
6. **专题号称 13 讲，实际只有 4 篇 lecture 文件存在**（`01-dpo.md`/`04-orpo.md`/`05-simpo.md`/`12-rainbowpo.md`），README 明确写"L02-L13 待续"。L12 结尾预告的"L13 Capstone"markdown 本身**不存在**——capstone 只以 `capstone_dpo_comparison.py` + notebook 的形式存在，没有配套的书面 lecture。
7. Lecture `12-rainbowpo.md` Slide 14 引用的"LLaMA-3 8B + UltraFeedback，SimPO AlpacaEval2 LC 44.7 vs DPO 40.4"等数字是**论文报告的数字**（Meng 2024, SimPO paper），不是本项目自己复现出来的结果——这一点在文档里必须清楚区分，不能在面试中把"读到的论文数字"和"自己跑出的数字"混为一谈。

---

## 6. 追问树（5 条链，每条 3-4 层）

### 追问 1 · 对比的可信度

**trigger**："这几个对齐算法（DPO/KTO/ORPO/SimPO）的对比在什么数据规模下做的，结论能推广到大模型吗？"

1. **Q**: 你的 6 方法对比具体是在什么规模的数据/模型上跑的？
   **A**: 是合成标量模拟——`benchmark()` 里 batch=16，每个样本只是一个随机初始化的标量 log-prob（`randn(16) - 5`），不是真实 tokenizer/真实模型输出，也没有接入 Anthropic-HH 真实数据（虽然代码 docstring 写了"mock"）。50 步单标量梯度下降，seed=0 固定。
2. **Q**: 那这个对比到底在验证什么，如果不是真实训练效果？
   **A**: 验证的是 6 个 loss 公式在**同一个更新场景下的方向性质**——比如 DPOP 的 hinge 项是否真的能防止 chosen log-prob 下降（`test_dpop_punishes_chosen_drop` 显式构造了 chosen 0.5→0.3 的反例验证这一点），SimPO 的 length normalize 是否让不同长度但等价 per-token 概率的序列得到相同 reward（`test_simpo_length_normalize`）。这是"定性验证 loss 设计意图"，不是"定量测出哪个方法更好"。
3. **Q**: 如果要让这个对比真正说明"哪个方法在大模型上更好"，还缺什么？
   **A**: 缺真实 tokenizer + 真实 transformer 前向（`get_log_probs_for_labels` 在 `dpo_minimal.py` 里已经写好，可以复用）、缺真实偏好数据集（Anthropic-HH 或 UltraFeedback）、缺训练到收敛后在下游任务（AlpacaEval/MT-Bench）上的评测——目前只有 `dpo_minimal.py` 一个变体具备"接真模型"的 `train()` 骨架，其余 5 个变体要先补上同样的 `train()` 才能做真实规模对比。
4. **Q**: 论文里 SimPO 44.7 vs DPO 40.4 这些数字，你自己核实过吗？
   **A**: 没有复现，这是 `lectures/12-rainbowpo.md` Slide 14 引用的 Meng 2024 SimPO 论文报告数字（LLaMA-3 8B + UltraFeedback），属于"读到的文献数字"，不是本项目自己跑出来的结果，面试中会明确区分这两类数字的来源。

**pitfall**：容易把"合成 mock 对比"和"真实训练对比"混着说，一旦被追问"你们用了多少条真实偏好数据"就会露馅——诚实的答案是"目前是 0 条真实数据的合成模拟，只有 DPO 有真实训练路径"。

---

### 追问 2 · 评估指标的局限

**trigger**："你怎么选的评估指标，这个指标有什么已知的局限？"

1. **Q**: capstone 里具体看哪几个指标？
   **A**: 三个：loss 曲线、reward margin（chosen−rejected 的 log-ratio 差，`margin_mean`）、chosen log-prob 漂移（`Δ chosen_logp = 末态 - 初态`，`print_table` 里 < 0 会标 "⚠️ 下降"）。
2. **Q**: margin 上升是否就等价于"模型变得更好"？
   **A**: 不等价，这正是 lecture `01-dpo.md` Slide 12 提到的"DPO 的奇怪现象"——margin 可以持续上升，但 chosen 的绝对概率反而下降（因为 loss 只追相对差，rejected 降得更快）。这也是本 capstone 专门加一列 `Δ chosen_logp` 的原因：**设计意图上**是想让 DPOP 在这一列上和 DPO 拉开差距。但诚实地说，因为 `VARIANTS["dpop"]` 在统一框架里和 `VARIANTS["dpo"]` 超参完全相同（见第 3.1 节），capstone 表格里 DPO 和 DPOP 这两列实际数字是一样的——这一列目前**没有真正展示出 DPOP 该展示的效果**，是设计目标和当前实现之间的一个具体落差。
3. **Q**: 那 chosen_logp 漂移这个指标本身有什么局限？
   **A**: 它是在合成标量场景里定义的（单一标量 log-prob 的变化），不是真实序列级的"生成质量"变化；在真实训练里，chosen 概率下降不一定代表输出变差——还需要人类评估或 AlpacaEval 这类下游胜率指标才能确认"下降"是否真的对应生成质量恶化。本项目没有做这一层验证。
4. **Q**: 如果让你重新设计这个 capstone 的评估指标，你会加什么？
   **A**: 会加"length 漂移"这个 lecture `12-rainbowpo.md` Slide 8 里提到但**代码里没有真正实现**的指标（capstone docstring 提到"length 漂移 (SimPO 优势 — 无漂移)"，但 `benchmark()` 实际代码没有对 response 长度做任何统计——mask 全是 1，序列长度 T=12 固定不变）；这是文档承诺但代码未交付的一个具体缺口。

**pitfall**：被问到"margin 涨是不是就是变好"时，如果直接说"是"会立刻暴露没理解 DPO chosen-prob-drop 这个专题反复强调的核心陷阱。

---

### 追问 3 · KTO 为什么被排除

**trigger**："RainbowPO 说统一了 7 个变体，为什么 capstone 只对比 6 个？"

1. **Q**: 具体是哪个变体没有进对比，为什么？
   **A**: KTO。`rainbowpo.py` 的 `VARIANTS["kto"]` 超参和 `VARIANTS["dpo"]` 完全一样（`use_ref=True, length_norm=False, loss_type="sigmoid", add_sft=False, beta=0.1`），`capstone_dpo_comparison.py` 里 `benchmark()` 遍历的列表是 `["dpo", "ipo", "orpo", "simpo", "cpo", "dpop"]`，显式不含 kto。
2. **Q**: 为什么统一框架处理不了 KTO？
   **A**: KTO 的数学形式本质上是单边的——每条数据是 `(response, is_desirable)`，不是 `(chosen, rejected)` pair。`kto_loss()` 的签名是 `(log_p_actor, log_p_ref, is_desirable, beta, lambda_d, lambda_u)`，和其余 6 个"pair 输入"的接口完全不同；`unified_po_loss()` 的参数硬编码了 `log_p_c_*`/`log_p_r_*` 两条序列，结构上没法表达"只有一条序列 + 一个二值标签"的场景。
3. **Q**: 那 `VARIANTS` 字典里保留 kto 这个键是为了什么？
   **A**: 说实话，这个键的存在更像是"占位/记录 7 个变体名字"，它的具体超参数值目前和 DPO 完全相同，跑 `unified_po_loss(cfg=VARIANTS["kto"], ...)` 会和跑 DPO 得到完全一样的数字——这个键**目前没有独立的数学意义**，只是没被删掉。
4. **Q**: 如果要让 KTO 真正接入这个统一框架，需要怎么改？
   **A**: 需要给 `POConfig`/`unified_po_loss` 增加一个"pair-free"分支或者单独的 `use_pair: bool` 维度，输入变成 `(log_p_actor, log_p_ref, is_desirable)` 而不是两条序列；这会把当前"4 维超参覆盖 6/7 个变体"的整洁性打破，是这个统一框架设计上需要权衡的地方——目前没有做这个扩展。

**pitfall**：容易顺口说"RainbowPO 统一了 7 个 DPO 变体"而不去检查 KTO 那一行其实是重复 DPO 的参数——考官如果追问"那 KTO 具体在统一公式里怎么表示单边偏好"，答不上来就露馅。

---

### 追问 4 · ORPO 的 log-odds 近似问题

**trigger**："ORPO 论文用 odds ratio，你的统一公式里 ORPO 是怎么处理的？和论文一致吗？"

1. **Q**: `orpo_minimal.py` 里 ORPO 的核心数学是什么？
   **A**: 用 `log_odds(p) = log(p/(1-p)) = log_p - log1p(-exp(log_p))` 替代 DPO 里的 `log π/π_ref`，因为它不依赖 ref model；loss = `L_SFT(chosen) + λ · (-logsigmoid(log_odds_chosen - log_odds_rejected))`。
2. **Q**: `rainbowpo.py` 的 unified 公式里 orpo 预设是怎么算 margin 的？和上面的 log-odds 一致吗？
   **A**: 不完全一致。`unified_po_loss` 对 orpo 预设（`use_ref=False`）算的是 `log_ratio_c = log_p_c_actor`（纯 log-prob，不做 odds 变换），然后 `margin = beta * (log_ratio_c - log_ratio_r)` 走 sigmoid loss + 加 SFT。也就是说统一框架捕捉了"无 ref + SFT 锚住"这个工程性质，但**没有复刻 log-odds 这个具体数学变换**。
3. **Q**: 这种近似会导致统一框架跑出来的"orpo"结果和真实 ORPO 论文的结果不一致吗？
   **A**: 会有偏差，但方向上仍然合理——因为 lecture `12-rainbowpo.md` Slide 13 本身承认"ORPO 不是严格 DPO 等价，而是另一种凸推 preference 的方式"，用普通 log-ratio 近似 log-odds 在两者数值都不太极端时行为相近（都是单调递增的 S 形），但极端概率下（p 接近 0 或 1）log-odds 会有更陡的梯度，普通 log-ratio 不会。测试代码里也没有专门验证这两种参数化在数值上的具体差多少。
4. **Q**: 如果要让统一公式精确复刻 log-odds，需要改什么？
   **A**: 需要在 `POConfig` 里再加一个变换维度（比如 `transform: "log_ratio" | "log_odds"`），在 `unified_po_loss` 里分支处理——目前 4 维超参没有覆盖这一维，这是统一框架"够用但不完备"的一个具体证据。

**pitfall**：如果只会说"RainbowPO 统一了 ORPO"而说不出"统一的是工程性质还是数学形式"，会被追问出这个近似的具体位置——这是检验"是否真的读过两份代码并对比过"的分水岭问题。

---

### 追问 5 · bug 修复的根因

**trigger**："commit log 里提到你修了一个 capstone 的 bug，具体是什么问题，怎么发现的？"

1. **Q**: 这个 bug 具体的报错是什么，出现在哪一步？
   **A**: `mock_step()` 在第二次被调用时会 `AttributeError`，因为上一步返回的字典里混入了 `"loss"`/`"margin"` 两个 Python float（来自 `.item()`），而函数开头对 `init_state` 每个值无条件调用 `.clone()`，float 没有这个方法。
2. **Q**: 为什么会有 float 混进本该全是 tensor 的字典？
   **A**: `mock_step` 的返回语句用字典 `|` 合并操作符，把"下一步要用的 state"（原始张量）和"本步的统计结果"（loss/margin 标量）拼进同一个字典返回；调用方 `benchmark()` 直接把这个返回值整体传给下一次 `mock_step(cfg, state)` 作为 `init_state`，没有先把统计字段摘出去。
3. **Q**: 这种 bug 说明设计上有什么问题，你是怎么修的？
   **A**: 根因是把"训练状态"（生命周期跨步）和"单步产出"（生命周期只在本步）放进了同一个字典，两者被同一段 clone/detach 逻辑无差别处理。修复方式是给 clone/detach 都加 `hasattr(v, "clone"/"detach")` 判断，非张量值原样透传——这是最小改动的 workaround，没有重构成两个字典分开传递。
4. **Q**: 这个 workaround 有没有潜在风险？如果以后往返回字典里加别的非张量字段呢？
   **A**: 有风险——`hasattr` 判断只是让当前这几个已知类型（float）不报错，但如果以后加一个需要真正 clone 语义的自定义对象、又没有 `.clone()` 方法，会被静默跳过而不是报错，可能掩盖新的 bug。更彻底的修法应该是把"state"和"metrics"分成两个独立字典返回，而不是继续往同一个字典里塞。**这一层反思代码里没有写，是我在复盘这个 bug 时的推断（推测，非文档明确记载）**。

**pitfall**：只说"加了个 hasattr 判断修好了"而说不出这个 bug 的根因（字典 `|` 合并导致类型污染）以及这个修法本身仍是 workaround 而非根治，会显得像是"看报错改到不报错为止"，缺乏对设计缺陷的反思深度。

---

## 附：核心数字来源速查

| 数字/断言 | 来源 |
|---|---|
| 6 变体最终 margin（DPO -0.001 … DPOP -0.001） | 实际执行 `notebooks/N1-po-variants-comparison.ipynb` 输出 |
| 14/14 测试通过，1.66s | 本次复核 `pytest learning/dpo-family/src/tests/` 实测 |
| bug 具体 diff（clone/detach hasattr 修复） | `git show 5f977cf -- learning/dpo-family/src/capstone_dpo_comparison.py` |
| 7 变体超参表 | `learning/dpo-family/src/rainbowpo.py` `VARIANTS` 字典 |
| SimPO 44.7 / DPO 40.4 等论文数字 | `learning/dpo-family/lectures/12-rainbowpo.md` Slide 14（转引 Meng 2024） |
| 只有 dpo_minimal.py 有 train() | 对 `src/` 全部 `*_minimal.py` 文件逐一确认 + grep `def train(` 全目录只命中 1 处 |
| 无 `*_trl.py` 文件 | glob `learning/dpo-family/src/*trl*` 无结果 |
