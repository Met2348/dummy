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
| 01 | DPO 基础与推导 | [01-dpo-foundations.md](01-dpo-foundations.md) | 5 | ✅ 已完成(已验证,含 `dpo_minimal.py` 真实训练复现字符级一致 + `resp_mask_c` 全文本掩码诚实标注 + `lectures/01-dpo.md` Slide 25 引用不存在的 `dpo_trl.py` 文档漂移发现;P/T/L/V/D 逐文件复核确认知识点 1 开头"预备知识"已完整定义 KL 散度/Bradley-Terry/配分函数,RLHF/KL/Bradley-Terry 地基问题在本文件已解决,追加 RLHF vs DPO 模型身份对比表格补一张结构可视化) |
| 02 | PO 变体家族 | [02-po-variant-family.md](02-po-variant-family.md) | 7 | ✅ 已完成(已验证,含 `inspect.signature` 澄清"KTO 仍需要 ref model,只省了配对"的常见误解 + DPOP 反例位级复现;P/T/L/V/D 逐文件复核确认正确复用 01 号文件的 KL/Bradley-Terry 定义,未发现需要改动之处) |
| 03 | RainbowPO 统一视角与 Capstone | [03-rainbowpo-and-capstone.md](03-rainbowpo-and-capstone.md) | 3 | ✅ 已完成(已验证,发现 `rainbowpo.py::unified_po_loss` 仅精确复现 `dpo` 一个配置——`dpop`/`kto` 配置字段和 `dpo` 逐字段相同、静默退化成纯 DPO,`ipo`/`orpo`/`cpo`/`simpo` 均与各自独立实现存在结构性数值偏差,capstone 50 步 benchmark 里 `dpop` 行与 `dpo` 行逐步位级相同;P/T/L/V/D 逐文件复核追加 `VARIANTS` 7 配置 × 4 轴取值对照表——原文只描述"和 lecture 表格核对一致"未把取值本身列出,已用 `.venv` 现场核对表格与源码逐字段一致) |
| 04 | 进阶深度追加:4 个多级追问链案例 | [04-advanced-interview-depth.md](04-advanced-interview-depth.md) | 4案例(不计入15) | ✅ 已完成(已验证,8/8代码块独立通过;基于真实WebSearch调研的5条追问轴线撰写——①RainbowPO"一个函数统一7个变体"真实性验证(核心案例,03类发现的延伸;独立复验阶段用结构化字段比对证明dpop/kto与dpo配置逐字段相同是与随机种子无关的事实,用cpo_minimal.py源码逐行推导出"unified_cpo恒等于2×real_cpo"这条代数恒等式而非数值巧合,并用第4组全新种子77777/B=11/T=13复现全部匹配模式)、②DPO→IPO→KTO→ORPO/SimPO/CPO→DPOP方案批判迭代链(核心案例;独立复验阶段用不同学习率0.02+2000步复现DPO持续上涨/IPO精确收敛的动态,并从源码推导出DPO梯度`-β·sigmoid(-βh)`永不为零、IPO梯度`2(h-target)`恰为零的解析证明,期间发现并纠正了自己"IPO与DPO共享beta缩放约定"的错误假设——ipo_minimal.py的h是原始log-ratio差、未被beta预先缩放,和dpo_minimal.py的margin=beta·h是不同的约定)、③给定约束选算法决策依据追问(核心案例)、④真实训练vs纯数值demo规模递增(核心案例,含真实GPT-2逐条vs批量forward耗时实测+真实hh-rlhf数据集多轮对话mask稀释统计);P/T/L/V/D 逐文件复核未发现需要改动之处) |
| 05 | 手把手实战:从零搭一个迷你 DPO 训练循环 | [05-build-a-mini-dpo-loop.md](05-build-a-mini-dpo-loop.md) | 4阶段(不计入15) | ✅ 已完成(已验证,5/5代码块独立通过;复用 01 号文件知识点 2 的 `dpo_loss` 公式与实现不做任何改动,新增"policy logits 矩阵 → log_softmax → log-prob"这一步补上 01-04 号文件里始终被跳过的一环,用 5x8=40 个可训练标量 + 5 条 toy 偏好对跑通 前向loss→backward→手写参数更新→循环训练→组装成可复用训练器 全流程;阶段2现场发现反直觉事实——同一 prompt 行内未被选中比较的候选,梯度精确为0.0(不是很小),和01号文件Z(x)精确抵消是同一代数结构;阶段4现场发现5条独立行的偏好对margin轨迹精确重合到小数点后四位,一度怀疑是bug,现场推导后确认是"每条pair占一整行、互不重叠"这个玩具简化的结构性产物而非DPO本身性质;附加实验现场验证两条共享同一行的矛盾偏好对margin精确锁死在0、连续50步不动,为"标注矛盾会拖慢/抵消训练信号"给出一个精确可复现的最小反例) |

**预计合计:约 15 个知识点,3 篇 + 1 篇进阶深度追加(4 个案例,不计入 15)+ 1 篇教程体试点(4 阶段,不计入 15),全部完成并独立验证。**

**关于 05 类的方法论说明:** 这是 [dsa-deep-dive/21-build-a-mini-search-engine.md](../dsa-deep-dive/21-build-a-mini-search-engine.md) 首次验证过的"教程体"内容形态在本系列的落地——01-03 号文件是"公式与代码逐项对照"/"变体横向对比",04 号文件是"旁观一条多级追问链",三者都是**读者查阅或旁观**;05 号文件要的是**读者动手**:从空文件开始一步步敲代码、每段代码在 `.venv` 里真实跑出数字之后才誊写成 `assert`,最终独立组装出一个真实能跑起来的迷你 DPO 训练循环。**明确不计入"约 15 个知识点"的统计**,和 04 类"进阶深度追加"一样是独立于知识点计数之外的一类内容。选择 DPO 训练循环作为落地对象,是因为 01 号文件知识点 2/4 之间恰好留了一块空白——`dpo_loss` 公式本身、以及一整条真实 GPT-2 forward 链路都已经讲过,但"log-prob 这个 loss 的输入到底是怎么从一个可训练参数里长出来的"这一步在 01-04 号文件里始终是给定的,05 号文件用一个 5x8 的玩具 logits 矩阵把这一步补齐。撰写时额外核实过 `learning/dpo-family/src/capstone_dpo_comparison.py`(03 号文件知识点 2 的对象)已经有一个类似的"多轮 mock 训练循环",05 号文件在正文里明确写清楚两者的简化方式不同(capstone 直接把 log-prob 当自由参数,05 号文件从 logits 矩阵出发经 `log_softmax` 推出 log-prob),避免重复而不作说明。

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
