# 对齐算法深挖 —— 路线图与进度表

> 目标:约 15 个偏好优化(Preference Optimization)算法知识点,由浅入深,深度对标 [torch-deep-dive/](../torch-deep-dive/00-roadmap.md)/[huggingface-deep-dive/](../huggingface-deep-dive/00-roadmap.md)(面试二三四面深度,不是"这个函数怎么调")。
> 定位:仓库"专题精读系列"第 3 条,直接对应 `learning/dpo-family/` 这一个专题模块——8 种 PO(Preference Optimization)算法(DPO/IPO/KTO/ORPO/SimPO/CPO/DPOP)+ 统一视角 RainbowPO,合计 9 个 `src/*.py` 文件,已逐文件读过(含行数、每个文件实现什么、真训练 vs 纯数值模拟的精确分界、已有的 pytest 断言),本系列直接复用那批调研结果。

---

## 和 `learning/dpo-family/` 的关系(差异化声明,必须先读)

`learning/dpo-family/` 本身的 `paper/guide_01_direct_preference_optimization.md`(426 行)已经把 DPO 论文的数学推导讲得很深——本系列**不重复那份推导**,而是聚焦"这些算法在代码里长什么样、8 种变体之间具体差在哪一行公式、以及怎么用代码交叉验证这些差异"。每个知识点从"最笨的想法"讲起(比如先问"RLHF 三段式那么麻烦,能不能把 reward model 直接消掉",再引出 DPO 的代换技巧),额外多两块:**底层机制/为什么这样设计** + **面试怎么问+追问链**。

**一个重要的诚实标注**:`learning/dpo-family/` 目前只有 4 篇 lecture(`01-dpo.md`/`04-orpo.md`/`05-simpo.md`/`12-rainbowpo.md`),覆盖 8 个算法里的 DPO/ORPO/SimPO/RainbowPO 四个;IPO/KTO/CPO/DPOP 这四个算法**只有 `src/` 代码,没有对应 lecture**。本系列撰写这四个算法的知识点时必须直接从代码本身出发(读函数实现、读 docstring、读论文原意),不能假设存在配套 lecture 可以转述。

---

## 环境声明

运行环境:仓库根目录 `.venv`(Windows 原生,Python 3.13)。**全系列以纯 CPU 数值模拟为主**——`learning/dpo-family/src/` 下 9 个文件里,**只有 `dpo_minimal.py` 一个是真训练脚本**(真实加载 GPT-2 actor+ref 模型 + `Anthropic/hh-rlhf` 数据集,argparse 可跑,`runbook.yaml` 标注 `gpu: true`),其余 8 个(`ipo_minimal.py`/`kto_minimal.py`/`orpo_minimal.py`/`simpo_minimal.py`/`cpo_minimal.py`/`dpop_minimal.py`/`rainbowpo.py`/`capstone_dpo_comparison.py`)全部是纯 torch 张量级数值 demo,不加载任何模型、不需要 GPU、不需要网络、秒级跑完。本系列的"可运行例子"以这 8 个纯数值文件为主,`dpo_minimal.py` 的真训练部分作为知识点 4"如实标注为可选进阶验证",不强制要求读者跑一次真训练。

已用 `grep -r "trl" learning/dpo-family/src/*.py` 核实:**全部 9 个文件零 `trl` 库依赖**,全部是手写的张量级 loss 实现——这是仓库审计记录里明确验证过的架构选择,本系列会展开讲"为什么手写而不直接用 trl 的 `DPOTrainer`"这个工程决策。

---

## 知识点结构模板(七步,与 torch-deep-dive/huggingface-deep-dive 完全一致)

1. **签名/是什么** 2. **一句话** 3. **底层机制/为什么这样设计** 4. **AI 研究/工程场景** 5. **可运行例子**(带 assert,真在 `.venv` 里跑过) 6. **面试怎么问 + 追问链** 7. **常见坑**

---

## 进度表

| # | 分类 | 文件 | 知识点数(约) | 状态 |
|---|------|------|-----------|------|
| 01 | DPO 基础与推导 | [01-dpo-foundations.md](01-dpo-foundations.md) | 5 | ✅ 已完成(已验证,含 `dpo_minimal.py` 真实训练复现字符级一致 + `resp_mask_c` 全文本掩码诚实标注 + `lectures/01-dpo.md` Slide 25 引用不存在的 `dpo_trl.py` 文档漂移发现) |
| 02 | PO 变体家族 | [02-po-variant-family.md](02-po-variant-family.md) | 7 | ✅ 已完成(已验证,含 `inspect.signature` 澄清"KTO 仍需要 ref model,只省了配对"的常见误解 + DPOP 反例位级复现) |
| 03 | RainbowPO 统一视角与 Capstone | [03-rainbowpo-and-capstone.md](03-rainbowpo-and-capstone.md) | 3 | ✅ 已完成(已验证,发现 `rainbowpo.py::unified_po_loss` 仅精确复现 `dpo` 一个配置——`dpop`/`kto` 配置字段和 `dpo` 逐字段相同、静默退化成纯 DPO,`ipo`/`orpo`/`cpo`/`simpo` 均与各自独立实现存在结构性数值偏差,capstone 50 步 benchmark 里 `dpop` 行与 `dpo` 行逐步位级相同) |

**预计合计:约 15 个知识点。**

---

## 明细(源码路径,撰写时逐一核实文件路径/行号仍然准确)

### 01 DPO 基础与推导(源:`learning/dpo-family/src/dpo_minimal.py`,配套 `lectures/01-dpo.md`398行)
1. 从 RLHF/PPO+KL 闭式解到 Bradley-Terry 代换的推导(呼应仓库已有的 `rl-foundations`/`rlhf-classic`,不重复 PPO 本身,只讲"怎么绕开显式 reward model")
2. 最终 DPO loss 公式与 `dpo_minimal.py::dpo_loss` 实现对照,逐项对应代码里的变量名
3. β 超参的作用(控制偏离参考模型的程度)
4. **`dpo_minimal.py` 是全系列唯一真训练脚本**——真实 GPT-2 actor+ref+`Anthropic/hh-rlhf`(或降级 dummy pair,仍真训练),argparse 可跑,如实标注为"可选进阶验证"
5. pytest 断言:零 margin 时 loss 精确等于 `log(2)`(来自 `src/tests/test_dpo_loss_equivalence.py`),这是验证 loss 实现正确性最基础的一条边界检查

### 02 PO 变体家族(源:`learning/dpo-family/src/{ipo,kto,orpo,simpo,cpo,dpop}_minimal.py`)
1. IPO(`ipo_minimal.py`)—— 平方损失,抗过度自信,和 DPO 在大 margin 下的行为差异对比
2. KTO(`kto_minimal.py`)—— 单边(不需要 chosen/rejected 配对)的偏好损失
3. ORPO(`orpo_minimal.py`)—— odds-ratio 免参考模型(不需要加载单独的 reference 模型),配套 `lectures/04-orpo.md`
4. SimPO(`simpo_minimal.py`)—— 长度归一化免参考模型,配套 `lectures/05-simpo.md`
5. CPO(`cpo_minimal.py`)—— 对比 margin + SFT 项
6. **DPOP**(`dpop_minimal.py`)—— chosen 概率下降现象的反例构造与 hinge 修复,已有 pytest 断言 `test_dpop_punishes_chosen_drop` 验证 `L_dpop > L_dpo`(在同一个"chosen 概率反而下降"的反例场景下)
7. 8 种算法一张表对比(何时选哪个:需不需要参考模型、需不需要配对数据、计算开销差异)

### 03 RainbowPO 统一视角与 Capstone(源:`learning/dpo-family/src/{rainbowpo,capstone_dpo_comparison}.py`,配套 `lectures/12-rainbowpo.md`254行)
1. RainbowPO 的 4 轴统一框架(`use_ref`/`length_norm`/`loss_type`/`add_sft`)——把前面 6-7 种变体表示成这 4 个开关的不同组合;已有 pytest 断言 `test_rainbowpo_dpo_matches`(RainbowPO 的 `"dpo"` 配置数值精确匹配独立 `dpo_loss` 到 1e-5)
2. Capstone:6 变体对比(`capstone_dpo_comparison.py`,50 步 mock 梯度下降),可视化 DPOP 修复的正是 DPO 会出现的"chosen logp 下降"现象——两个知识点用的是同一个反例,这里从"整个训练轨迹"的角度重新看待知识点 02-6 讲过的静态断言
3. Zero trl import 的架构选择——为什么全系列 9 个文件手写实现,而不是直接调 `trl.DPOTrainer`/`KTOTrainer` 等现成封装;呼应仓库 runbook 审计记录里"避免踩 trl 1.5.x 版本漂移坑"的判断

---

## 撰写与验证纪律

- 每个知识点的可运行例子必须在仓库根目录 `.venv` 真实跑通;涉及 `dpo_minimal.py` 真训练的部分,明确标注为可选进阶验证,不强制要求。
- IPO/KTO/CPO/DPOP 四个知识点没有配套 lecture,必须直接从 `src/*.py` 源码和函数 docstring 出发撰写,不能假设存在可参照的中文讲解。
- 每写完一批,在本文件进度表如实更新状态(⬜ 待撰写 → 🔧 撰写中 → ✅ 已完成,验证通过才标"已完成")。

---

*创建:2026-07-12*
