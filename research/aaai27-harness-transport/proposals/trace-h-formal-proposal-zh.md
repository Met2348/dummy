# TRACE-H 正式 Proposal：Harness 可迁移性的预测、分解与风险控制

> **历史版本：** 本文的 patch-effect predictor method center 已由新的 design-method Proposal [向未见 LLM Executor 运输可执行 Harness 控制策略](trace-h-policy-transport-proposal-zh.md)取代。本文只保留第一、二轮文献边界、原子契约和 sealed prediction 历史；当前方法、PK 与主指标以新 Proposal 为准。

- **项目代号：** TRACE-H（Trigger-Rate-Adjusted Conditional Effects for Harnesses）
- **中文简称：** 面向模型升级的 Harness 运输科学
- **目标会议：** AAAI-27 主会
- **版本日期：** 2026-07-11，第二轮全文碰撞复核版
- **状态：** 系统级主张、原子化验证；先运行 collision-aware 72 小时证伪实验
- **替代对象：** 旧版 FORECAST-H 通用特征预测方案
- **研究愿景：** 把 harness 从与单一模型绑定的经验工程，提升为可测量、可预测迁移、可审计和可控制风险的独立系统层。
- **核心任务：** 建立 Harness Transportability 问题、需求-响应运输律和密封评测协议，并以事件触发型原子 patch 为可识别实验单元，在看不到目标补丁结果时预测其净效应、选择最优 patch/none 或弃权。

## 摘要

当基础模型升级时，现有 agent harness 往往被整体照搬或重新调参，但模型与 harness 共同决定系统行为：同一个重试、纠错或终止机制，在弱模型上可能补救失败，在强模型上却可能制造冗余、成本甚至回退。TRACE-H 研究的不是某个提示技巧，而是一个更一般的问题：能否把 harness 从与单一模型绑定的经验工程，提升为可测量、可预测迁移、可进行风险控制的独立系统层？

我们提出 Harness Transportability 问题。给定若干来源模型上的运行记录和一个从未执行过补丁的目标模型，系统必须在部署前预测每个 harness patch 在目标模型上的有符号效应及不确定性，并决定复用、拒绝或暂缓部署。核心假说是“需求-响应运输律”：补丁的目标效应不由模型规模或来源平均收益单独决定，而由目标 baseline 轨迹中可被该补丁处理的机会，与来源模型上该补丁在相同失败状态下的条件补救和伤害概率共同决定。通用 headroom 只能说明目标可能受益，patch-specific opportunity × response 才决定该用哪一个补丁。

TRACE-H 将重试、错误恢复和终止检查实现为满足 dormant-until-trigger 契约的原子补丁；触发前与 baseline 完全一致，触发后仅进行一次有界干预。方法从来源配对实验学习条件响应，从目标 baseline 轨迹估计机会分布，输出每个补丁的效应、区间及 patch/none/abstain 决策。实验采用三个来源模型、一个完全密封的未见目标模型、三类补丁和三十六个离线任务，通过预测误差、符号准确率、区间覆盖、负迁移率和选择后悔检验方法。

论文预期贡献不是“再发现 harness 有影响”，而是系统建立 harness 迁移的预测任务、运输分解、密封评测协议和风险感知决策框架，并回答一个面向未来 agent 工程的核心问题：模型可以替换时，哪些控制逻辑仍应保留，哪些必须删除，能否在昂贵的目标侧试错之前做出可靠判断。若成立，TRACE-H 将把 harness 优化从逐模型反复搜索推进为可积累、可审计、可迁移的系统科学。

## 1. 一句话论文

> TRACE-H 建立 Harness Transportability：用目标模型“需要什么”与来源补丁“在何种状态下有效”组成需求-响应运输律，在任何目标补丁试跑之前预测应复用、拒绝还是暂缓部署哪些控制逻辑。

## 2. 为什么必须重构旧 idea

旧版 FORECAST-H 试图从模型接口特征、任务特征、运行预算和 intervention descriptor 学习一个通用层次模型。它有四个致命弱点：

1. **机制不清：** 特征与补丁效应之间缺少可检验的中间机制，容易被评价为小样本黑盒回归。
2. **自由度过多：** 四个模型无法支撑大量 model-feature interaction，复杂模型只会放大过拟合疑虑。
3. **范围过大：** 多领域、多补丁、多预算、校准、benchmark 和部署规则难以压进 AAAI 七页正文。
4. **近邻工作太密：** 绝对性能预测、完整 harness 比较、事件触发技能、跨模型技能复用、frozen selector 和 prospective edit manifest 都已有直接先例。
5. **宽泛预测主张已碰撞：** Metric Freedom 已经从目标 baseline runs 事前预测 skill utility；任何新方案必须证明 patch-specific 增量，而非重新证明 baseline behavior 有信息。

TRACE-H 的调整不是换一个回归器，也不是把论文缩成补丁选择器。原子 patch 是为了获得可识别的实验单元；更大的科学主张是建立 harness 的运输对象、运输规律、部署前预测任务和风险控制协议，使 harness 能像模型与数据一样成为可积累、可比较的系统层。

## 3. 文献边界与真正空白

### 3.1 全文复核后已经成立的近邻事实

- [Metric Freedom](../foundations/second-wave/papers/2604.01608.pdf) 已从目标模型无 skill 的 repeated baseline runs 计算 score-landscape rigidity，在任何 skill 执行前预测 bundled MAS-derived skill lift；主分析在 13 个 task-dataset-metric 点上报告 `r=-0.85`，并用相同 skill definitions 在 GPT-5.1 上复核 `r=-0.71`。它占据宽泛的 baseline-only a priori utility prediction，但不区分同一目标上的不同 patch，也没有 sealed held-out prediction set、signed-effect calibration 或 deployment regret。
- [AHE](../papers/2604.25850.pdf) 已在每轮运行前写 predicted fixes 与 at-risk regressions；全文的 regression precision/recall 仅 11.8%/11.1%，且是在同一模型、同一任务集上的 bundled evolution。因此“预先写下风险预测”已被占据，但跨未见模型的准确 effect forecast 尚未完成。
- [Self-Harness](../papers/2606.09498.pdf) 对每个 candidate edit 反复查询所谓 held-out split 作为接受门；这不是 sealed final test，反而明确约束本项目必须把 target patch outcomes 完全隔离到最终 unseal。
- [What Should a Skill Remember?](../papers/2606.09421.pdf) 已用 calibration/adaptation/held-out 划分形成 frozen selector，并在多个 agent backbones 上运行；通用 frozen skill selector 不能作为本文新颖性。
- [HarnessFix](../papers/2606.06324.pdf) 将 GPT-5 mini 上修复的 GAIA harness 原样迁移到四个其他模型，均取得正增益；冻结修复物跨模型 transfer 已有直接先例。
- [The Harness Effect](../foundations/second-wave/papers/2607.06906.pdf) 在同一 frozen production loop 的 6-model matrix 中得到 30 个改善、11 个持平、7 个回退，并报告 mean quality gain 与 baseline strength 的 `r=.99 (n=6)`；baseline capability 是必须击败的强零假设。
- [A Framework for Evaluating Agentic Skills at Scale](../foundations/second-wave/papers/2606.17819.pdf) 已在约 500 skills、1,000 tasks 和 19 agent-model configurations 上测量 individual skill delta；它提供大规模异质性证据，但没有预测器，且不同模型部分绑定不同 harness。
- [Harness-Bench](../foundations/papers/2605.27922-harness-bench.pdf) 已在 106 个离线任务、多个模型与完整 harness 配置上测量 configuration-level variation；作者明确不把结果解释为单一机制的因果分解。
- [SEAGym](../papers/2606.17546.pdf) 的中心贡献是 self-evolving agent evaluation environment。其 AHE 跨 backend ablation 报告整套 harness 在 DeepSeek、GLM 和 GPT-5.4 间的非对称迁移与符号反转；作者读取 source trajectories 与 update artifacts 后，用评测轨迹是否继续暴露相似 dominant failure surface 作事后机制解释。它没有定义定量 failure-surface distance、原子 patch effect 或 sealed target forecast。因此 H-FS 假说不属于本文首创，但其前瞻、定量、原子化检验仍未完成。
- [LIFE-HARNESS](../papers/2605.22166.pdf) 从 Qwen3-4B-Instruct 轨迹演化七套环境专用四层 harness，冻结后复用于 17 个其他 backbone，在 126 个 model-environment 单元中改善 116 个；完整表仍包含持平与下降单元。它建立了强 cross-model transfer 先验，但所有目标都已实际运行整套 harness，且没有部署前 predictor 或 atomic decomposition。
- [HASP](../foundations/papers/2605.17734-hasp.pdf) 已把技能实现为带 `should_activate` 和 `intervene` 的可执行 Program Function，并分析触发频率与干预方式。
- [ContractSkill](../foundations/papers/2603.20340-contractskill.pdf) 已用前置条件、后置条件、恢复规则和局部修复构造可执行技能，并报告有限范围内的跨模型复用。
- [ToolBench-X](../foundations/papers/2606.25819-toolbench-x.pdf) 已系统注入五类可恢复工具风险，并显示定向诊断提示可在多个模型上恢复大量失败。
- [The Verifier Tax](../foundations/papers/2603.19328-verifier-tax.pdf) 已测量两个模型上的 verifier 触发、恢复率、安全收益和计算代价。
- [PACE](../reading-cards/2607.02032.md) 已从廉价 atomic capability probes 预测 held-out model 的绝对 agent benchmark 分数。
- [Agent Psychometrics](../reading-cards/2604.00594.md) 已预测未见的 model-scaffold 组合，但模型与 scaffold 身份分别在训练中出现过。
- [More Is Not Always Better](../reading-cards/2605.05716.md) 已证明 scaffold component 的交互、任务依赖和模型规模依赖。

因此，下列说法不能作为本文贡献：harness 很重要、补丁会触发、环境专用 harness 可广泛跨模型复用、迁移可能为负、模型与 harness 存在交互、failure-surface alignment 可能影响迁移、应在 held-out model 上验证、可以从 target baseline runs 事前判断 generic skill utility、可以冻结 selector、可以在 edit 前写 risk manifest。第一轮边界见[全文碰撞复核](../foundations/notes/fulltext-collision-reassessment-zh.md)，第二轮直接碰撞见[Metric Freedom 后的全文审计](../foundations/notes/second-wave-fulltext-collision-audit-zh.md)。

### 3.2 仍然存在的精确空白

本项目当前未发现工作同时完成以下更窄的前瞻预测协议：

1. 研究对象是一组版本化、事件触发、具有 dormant 契约的 frozen atomic patches；
2. 在同一目标 task/metric 的 generic headroom 下，分别预测每个 patch 相对同一 baseline 的 signed magnitude；
3. 目标模型上的全部 patch outcomes 在方法、阈值、patch 选择与预测文件完成前完全密封；
4. 目标侧只允许 repeated baseline runs、baseline telemetry 和预声明接口信息；
5. 把 Metric Freedom、baseline strength、failure-surface similarity 与 opportunity-only 作为强基线，再检验 patch-specific conditional response 的增量；
6. 最终评价 interval/calibration、negative transfer、top-1 patch choice、choose-none 与 decision regret，而不只是预测相关系数。

最终论文不得声称“首个 prospective skill-utility predictor”。即使结果成功，也优先直接陈述严格设定与可验证贡献，并说明“本次检索未发现 exact match”；不依赖 first-like 表述支撑 novelty。提交前仍须再次检索。

## 4. 研究问题

### RQ1：Generic headroom 是否已经足够？

Metric Freedom、baseline success/cost 与通用 adherence proxy 是否已经能预测 target patch effect 和最优 patch，从而使 patch-specific 机制没有必要？

### RQ2：Patch opportunity 与 conditional response 各提供多少增量？

控制 generic headroom 后，目标模型暴露不同 patch-addressable opportunity 能否解释 effect shift；按触发类型与 baseline 状态条件化后的来源 rescue/harm 是否进一步可迁移？

### RQ3：能否在 patch outcome 密封时预测目标效应？

只使用目标 baseline runs 构造的 `F`、失效面和 patch opportunities，是否能在真正密封的 held-out model 上提高 patch-specific effect MAE、符号判断、区间覆盖与负迁移识别？

### RQ4：预测是否改善实际部署决策？

基于不确定性的 `choose none / choose patch / abstain` 规则，是否比 always-reuse、source-copy、Metric Freedom-informed、capability-nearest、failure-surface-nearest 和 opportunity-only 减少 patch-selection regret？

## 5. 形式化定义

### 5.1 实验对象

设：

- `m` 为基础模型；
- `i` 为任务；
- `h0` 为固定 baseline harness；
- `pj` 为第 `j` 个原子 patch；
- `Y(m,i,pj)` 为任务成功指标；
- `C(m,i,pj)` 为归一化资源成本；
- `U = Y - lambda * C` 为预声明净效用，`lambda` 在看到目标 patch outcome 前冻结。

任务级有符号效应为：

```text
D(m,i,j) = U(m,i,pj) - U(m,i,h0)
```

模型或任务类别 `d` 上的平均效应为：

```text
Delta(m,d,j) = E_i[D(m,i,j) | i in d]
```

主论文首先报告成功率效应 `Delta_Y`，成本调整效用作为共同预声明的次要结果，避免用事后选择的 `lambda` 改写结论。

目标模型上的部署动作不是分别对三个 patch 做互不相干的二元判断，而是：

```text
a*(m,d) in {none, p1, p2, p3, abstain}
```

对非 abstain cell，patch-choice regret 定义为真实最优动作效用与所选动作效用之差：

```text
R_choice(m,d) = max_a Delta_U(m,d,a) - Delta_U(m,d,a_hat)
```

其中 `Delta_U(..., none)=0`。这使 Metric Freedom 的 generic “是否有 headroom”与 TRACE-H 的 patch-specific “该选哪一个”在同一决策量上可直接比较。

### 5.2 Dormant-until-trigger 契约

每个 patch 必须满足：

1. 触发谓词 `g_j(state, proposed_action)` 在运行前确定；
2. 第一次触发前，`pj` 与 `h0` 的输入、状态转移和可见信息完全一致；
3. 不触发时，patch 不添加 token、tool call、延迟或隐藏状态；
4. 触发后的干预类型、最大次数和最大额外预算固定；
5. 触发器不能读取 ground-truth answer、最终 oracle score 或目标 patch outcome；
6. 任何 contract violation 都单独报告，不能当作有效实验数据。

在共享随机种子/随机流或确定性执行下，若 baseline 轨迹从未满足触发条件，则 patched trajectory 也不会在此前自行分叉，因此该配对任务上的效应应为零。这同时构成一个可自动测试的实现不变量。`temperature=0` 本身不保证 provider 级确定性：若 endpoint 不支持 seed 或 replay，只能通过预声明重复估计分布层效应，不能把两次独立轨迹称为精确 paired potential outcomes。

### 5.3 触发条件化分解

令 `K_j` 表示 baseline 轨迹中 patch `j` 的第一次机会类型；`K_j=0` 表示从不触发，`K_j=k>0` 表示进入预声明的第 `k` 类失效状态。由全期望公式及 dormant 契约：

```text
Delta(m,d,j) = sum_k pi(m,d,j,k) * rho(m,d,j,k)
```

其中：

```text
pi(m,d,j,k)  = P(K_j = k | m, d)
rho(m,d,j,k) = E[D(m,i,j) | K_j = k, m, d]
```

且理论上 `rho(..., k=0)=0`。对于二元成功结果，条件效应还可以写成：

```text
rho_Y = P(baseline fail, patch pass | K=k)
      - P(baseline pass, patch fail | K=k)
```

前一项是 rescue，后一项是 harm。这比只报告平均 delta 更直接地暴露补丁为什么有用或有害。

### 5.4 从近邻解释到可证伪假说层级

TRACE-H 不把所有内容都包装成一个新假说，而是明确区分四层：

1. **H-MF，generic headroom 假说：** score-landscape rigidity 与 baseline strength 已能预测 skill/harness 的通用收益空间。该层来自 Metric Freedom 与 The Harness Effect，不是本文首创。
2. **H-FS，失效面对齐假说：** 当来源 harness edit 所覆盖的失败模式也出现在目标 baseline 轨迹中时，迁移更可能成功；模式转移时收益缩小或反转。该定性、bundle-level、事后解释来自 SEAGym，不是本文首创。
3. **H-OPP，机会标准化假说：** 对 dormant atomic patch，目标模型进入 patch-addressable stratum 的概率 `pi_target` 能解释并预测一部分总体 effect shift。
4. **H-RESP，条件响应可迁移假说：** 在足够具体、预先定义的第一次触发类型内，patch-specific rescue/harm profile 比未条件化总体效应更稳定；它与 `pi_target` 组合后应在 H-MF/H-FS/H-OPP 之上改善 patch choice。

TRACE-H 的核心方法增量是 patch-specific H-RESP，而不是重新命名 H-MF 或 H-FS。目标预测为：

```text
hat_Delta(target,d,j)
  = sum_k hat_pi(target,d,j,k) * hat_rho(source,d,j,k)
```

`hat_pi` 只来自目标 baseline 轨迹，`hat_rho` 只来自来源模型的 paired patch experiments。该稳定性是假设和实验对象，不是自动成立的因果识别定理。实验必须分别报告 H-MF、H-FS、H-OPP 与 H-RESP 的增量，不能只证明“baseline behavior 或相似失效面可能有用”。

## 6. TRACE-H 方法

### 6.1 目标 Baseline 的双重表示

第一部分是 generic headroom control。严格按照 Metric Freedom 的信息预算，在预声明 probe tasks 上对目标 baseline 做 repeated runs，计算：

- `F_out = 1-r_M(output distance, score distance)`，作为正文主控制；
- baseline success、cost、trajectory length；
- 可从 baseline 直接观察的 skill/harness activation 与 instruction-adherence proxy；
- `F_trace` 只作附录 sensitivity，避免正文依赖昂贵 embedding。

这些变量只回答“该模型/metric 是否存在一般结构化干预空间”，不能读取 patch outcome，也不能因 patch 身份而变化。按原文 mixed-question rule 计算的 `MF-faithful` 与不排除预声明 probe 的 `MF-all` 同时报告；若 probe 全成/全败导致退化，只能标记 missing/degenerate，不能换任务。任何 TRACE-H 增量都必须发生在控制这些 generic signals 之后。

第二部分是 patch-specific failure surface。对每个目标 baseline 任务和 patch，记录：

- 是否存在触发机会；
- 第一次触发的离散类型；
- 第一次触发发生的 step bin：early / middle / late；
- 触发时剩余预算 bin：low / adequate；
- baseline 最终成功与否；
- 同类事件总次数，仅用于次要分析。

主模型只使用预声明的粗粒度类型，避免在 36 个任务上制造高维特征。更细粒度错误文本 embedding 仅可作为探索性附录，不能替代主结论。`F` 对同一 task/metric 的所有 patch 相同，而 failure surface 随 patch trigger 改变；这正是 generic 与 patch-specific 信息的可检验分界。

### 6.2 来源条件效应估计

对每个 `(patch, trigger_type, baseline_status)`，将配对结果编码为 rescue、harm 或 unchanged。采用预先冻结的对称 `Dirichlet(1/2, 1/2, 1/2)` 多项平滑，得到条件概率和 `rho`；不使用深网络或高维 Bayesian hierarchy。

不确定性与跨来源敏感性分开处理：

1. 在任务类别内以任务为单位做 cluster bootstrap，每次重新估计 `pi`、`rho` 和目标 `Delta`；
2. 三个来源模型视为固定设计点，不用 `n=3` 的模型 bootstrap 冒充模型总体不确定性；
3. 通过三次 leave-one-source-model-out 和逐来源删除报告有限来源敏感性；
4. percentile interval 只解释为本任务集上的经验区间，coverage 只作描述性审计。

当某个 trigger stratum 在来源数据中样本不足时，按预声明层级回退：

```text
type x baseline_status x depth
  -> type x baseline_status
  -> type
  -> patch-level pooled effect
```

回退顺序和最小样本阈值在目标 unseal 前固定。

### 6.3 部署规则

定义实际业务最小有意义效应 `delta_min` 与预声明最优概率阈值 `tau_choice`。对每个目标 category：

1. 对每个 patch 计算 `hat_Delta_U(j)` 与 bootstrap joint draws；`UCB_j < 0` 的 patch 标记为 reject。
2. 令 `j* = argmax_j E[hat_Delta_U(j)]`，并计算 `P(j* = argmax_a Delta_U(a))`，其中 `a` 包含 `none`。
3. **choose patch `j*`：** `LCB_j* > delta_min` 且最优概率不低于 `tau_choice`。
4. **choose none：** 所有 patch 的 `UCB_j <= delta_min`。
5. **abstain：** 其余情况，交给目标特定调试或小规模 A/B，不声称自动 retune。

正文以 `choose patch / choose none / abstain` 为形式化动作；reject 是单 patch 状态，不是独立部署动作。旧版“reuse/retune/reject”中的 retune 没有独立算法，因此不再作为论文贡献。

### 6.4 为什么不声称风险保证

[Conformal Risk Control](../foundations/papers/2024-conformal-risk-control.pdf) 依赖 exchangeable calibration losses 和单调有界损失；[Learn then Test](../foundations/papers/2021-learn-then-test.pdf) 也需要足够、可解释的 calibration samples。少量模型替换 cell 并不自然满足这些条件。

因此本文只报告经验 risk-coverage curve、bootstrap interval 和 sealed-target decision regret。除非后续数据与假设完整满足相应定理，否则禁止使用“distribution-free guarantee”或“risk-controlled deployment”字样。

## 7. 三个原子 Patch

### P1：错误规范化器 E-NORM

- **触发：** 工具、进程或结构化动作返回机器可识别的 parse/schema/exit error。
- **干预：** 保留原始错误，同时追加固定 schema：错误类型、失败动作、可重试性、已消耗预算和下一步所需证据。
- **上限：** 每个独立错误最多规范化一次，不调用额外 LLM。
- **机制假设：** 主要改善“看见错误但没有正确诊断”的模型。

### P2：有界恢复器 R-BOUND

- **触发：** 同一规范化 recoverable error 第二次出现，或同一动作在无状态进展时重复。
- **干预：** 阻止一次立即重复，注入一个预声明的 bounded retry/fallback 选择，并保留原动作供审计。
- **上限：** 每个 episode 最多两次；不扩大工具权限。
- **机制假设：** 主要改善无效 continuation，但可能因额外步骤和错误 fallback 伤害已会自救的模型。

### P3：输出契约门 F-GATE

- **触发：** agent 尝试终止，但任务声明的非答案型输出契约未满足，例如必需文件不存在、格式不可解析、声明的测试命令未运行。
- **干预：** 拒绝终止一次，只返回缺失 contract predicate，不返回正确答案或 oracle 内容。
- **上限：** 每个 episode 一次，额外 step 和 token 单独计费。
- **机制假设：** 将“合理推理但没有落地为可验证 artifact”的失效转为可修复机会；也可能造成 verifier tax。

HASP 和 ContractSkill 已占据“可执行、可触发、可修复技能”的贡献。本文把这些 patch 当作受控处理变量，而不是方法新颖性来源。

## 8. 实验设计

### 8.1 主任务集

主计划使用 [Harness-Bench 2.0](../foundations/code/harness-bench/) 的固定快照 `1025086a446653702b80cfb48babbeec35db6b2c`。该快照包含 106 个离线 sandbox tasks 和程序化 oracle，避免网络状态漂移。

从以下三个类别各预选 12 个任务，共 36 个：

1. workspace、tool use 与 multimodal operations；
2. software engineering 与 codebase maintenance；
3. data、BI 与 finance analytics。

选择只依据任务 manifest、工具可用性、平台兼容性和 oracle 可运行性，不能依据任何模型结果或 patch 激活率。每类再预声明 6 个 Metric Freedom probe tasks，共 18 个；该 probe 身份同样在模型运行前冻结。若任务需要当前环境无法提供的 GUI/专有程序，必须在首次模型运行前整体替换并记录原因。

### 8.2 Harness 与模型轴

- 固定一个开源 reference harness 和同一 tool surface；
- 只更换 base model，不比较七个完整 harness；
- 三个来源模型来自不同模型家族；
- 第四个模型作为 prospective target，直到 predictor、阈值和 prediction file 冻结前不运行任何 patch；
- target 可以运行预声明的 repeated baseline diagnostic probes，但这些运行不得包含 patch、candidate acceptance 或目标 intervention feedback；
- 精确模型 ID、provider、版本日期、temperature、reasoning budget 和 context limit 写入 immutable manifest；
- 主运行优先选择支持固定 seed/replay 的 endpoint，并把 seed 写入 immutable manifest；
- `temperature=0` 但不支持 seed/replay 的 endpoint 不能承担任务级 no-trigger equality 主张，只能进入带预声明重复的分布层 sensitivity analysis；
- provider request ID、服务版本和原始响应全部保留。

Harness-Bench 原论文比较完整 native configurations；本文故意固定 reference harness，只操纵三个 wrapper patch，以避免把整个框架差异误称为原子效应。

### 8.3 主矩阵

| 维度 | 取值 |
|---|---|
| 模型 | 3 source + 1 sealed target |
| 任务 | 36，三个类别各 12 |
| 条件 | baseline + P1 + P2 + P3 |
| Primary effect episodes | `4 x 36 x 4 = 576` |
| MF diagnostic repeats | 18 probe tasks 每题额外 5 次，共 `4 x 90 = 360` |
| 总 primary + diagnostic episodes | `576 + 360 = 936` |
| 目标 baseline | 36 个 primary baseline + 90 个 diagnostic repeats；不含任何 patch outcome |
| 选择性重复 | 基础设施失败全部重跑并标记；边界 cell 最多重复 20% |

来源模型运行完整矩阵。每个模型先有 36 个固定配置的 primary baseline；18 个预声明 probe tasks 再各运行 5 次独立 diagnostic variants，与原 run 合计 `n=6`，用于忠实构造 Metric Freedom。diagnostic variants 只允许采用预声明 diversity prior，不参与 patch paired effect，也不能替代 primary baseline。每模型因此有 126 个 baseline/diagnostic episodes 与 108 个 patch episodes，共 234 个；四模型总计约 936 个。目标模型在这些 baseline-only 数据上生成 108 个 task-patch 失效面记录和 9 个 category-patch 效应预测；提交 seal 后才运行 108 个 target patch episodes。

这是一种 prospective transductive setting：允许看见目标任务的 baseline 行为，但看不到同一任务上的 patch outcome。论文必须清楚说明这一点，不能含糊地称为零目标交互。

### 8.4 验证结构

1. **来源内部验证：** 三次 leave-one-source-model-out；所有 trigger taxonomy、回退层级、Metric Freedom 缩放、generic baseline 和 `delta_min` 只用两模型训练、第三模型验证。
2. **冻结：** 根据来源验证一次性选择 TRACE-H 版本、Metric Freedom-informed baselines、部署阈值和 patch-choice policy。
3. **密封目标：** 写入目标 task-level 预测、聚合效应区间与 choose patch/none/abstain 决策，提交 commit hash。
4. **目标 audit：** 运行冻结的三个 patch，不再改 trigger、patch 或模型。
5. **最终审计：** 先报告 prospective target，再报告把全部四模型纳入的 retrospective sensitivity analysis。

### 8.5 与已发表 predictor/transfer 矩阵的外部一致性审计

Metric Freedom、SEAGym Table 28、LIFE-HARNESS Table 8 与 `A Framework for Evaluating Agentic Skills at Scale` 的公开矩阵不能直接冒充 TRACE-H 的训练或测试数据：前三者没有本项目的原子 patch/first-trigger telemetry，后一数据的模型与 harness 部分绑定。论文仍应把它们作为外部现象约束：

1. TRACE-H 的引言必须承认 Metric Freedom 已做 generic a priori utility prediction；
2. 必须同时解释 LIFE-HARNESS 的广泛正迁移与 SEAGym 的不对称/反号迁移，不能只挑负例制造问题；
3. source matrix 应检查“高 transfer prior 下的少数例外识别”，而不只报告平均 gain；
4. 若主实验只得到普遍正效应或完全由 `F`/baseline strength 解释，应承认结果支持已有工作，不能强行声称 patch-specific predictor；
5. 若出现反号 case，应比较 generic headroom、opportunity shift 与 post-trigger response shift 三层解释。

这是 narrative-level external audit，不冒充跨论文可比的 meta-analysis。

## 9. 假设与可证伪预测

### H1：Dormant 契约成立

没有 baseline trigger opportunity 的任务中，baseline 与 patch 的结果、tool sequence 和资源使用应一致。任何非基础设施差异均计为 contract violation。

### H2：Generic headroom 有效但不充分

Metric Freedom 与 baseline strength 应能解释一部分总体收益空间，但不能在同一 target category 内稳定区分三个 patch 的排序与符号。若它们已经足够，TRACE-H 的 patch-specific 主张失败。

### H3：失效机会解释 patch-specific 异质性

目标/held-out model 的 `pi_j` 变化应在 H-MF 控制后继续解释 raw patch effect；只用 generic `F`、baseline success 或 source mean 的模型不能达到相同 patch-ranking 表现。

### H4：条件效应比总体效应稳定

在来源 leave-one-model-out 中，trigger-conditioned `rho_j` 的预测误差应低于直接复制 source aggregate delta 与 opportunity-only。若条件化只增加方差而不提高预测，则 H-RESP 失败。

### H5：TRACE-H 改善密封目标预测

TRACE-H 在 task-level expected delta、9 个 category-patch aggregate effects、显著效应符号和 patch ranking 上，应优于最强 Metric Freedom-informed/simple baseline，而不是只优于零预测。

### H6：选择性 patch choice 降低负迁移

随着 coverage 降低，所选 patch 的负迁移率和 choice regret 应改善；在预声明 coverage 点，TRACE-H 应优于 always-reuse、source-copy、MF-nearest、Source x MF、failure-surface-nearest 与 opportunity-only。

## 10. Baselines

### 必需基线

1. **Zero-effect / choose none：** 所有 target delta 预测为 0，始终不部署。
2. **Source mean / source sign：** 复制来源模型平均总体效应或多数符号。
3. **Baseline strength：** 只用 success、cost、trajectory length 与通用 adherence proxy；直接响应 The Harness Effect 的强零假设。
4. **MF-headroom：** 严格计算 target `F_out`；由 source LOMO 冻结阈值判断是否存在 generic intervention headroom，具体 patch 排序仍沿用 source mean。
5. **MF-nearest：** 用 `(F_out, baseline strength)` 找最近来源模型，复制该来源三个 patch 的 effects。
6. **Source x MF：** 以预声明、裁剪后的 `(1-F_target)/(1-F_source)` 对 source patch prior 缩放；缩放式与边界在 target unseal 前冻结。
7. **Nearest capability：** 用 baseline success/cost 找最近来源模型并复制其效应。
8. **Failure-surface-nearest：** 对每个 category-patch cell，计算目标与各来源模型 baseline first-trigger distribution 的总变差距离 `TV = 0.5 * sum_k |pi_target(k)-pi_source(k)|`，包含 `k=0` 的 no-trigger 质量；选择最近来源并复制其同一 cell 的 aggregate patch effect。
9. **Opportunity only：** 使用 `pi_target`，但所有非零 trigger type 共享 pooled conditional response。
10. **Generic ridge：** 使用 `F_out`、baseline strength 与 TRACE-H 可见的低维 aggregate features，但不施加 typed trigger-response 分解。
11. **TRACE-H：** patch-specific `pi_j x rho_j` 与预声明回退。
12. **Active-target probe oracle：** 允许极少量目标 patch probes 后校准，用作信息预算上界，不与 baseline-only 方法混为同级。
13. **Full oracle：** 用真实目标效应选择动作，只用于 regret 归一化。

基线形成预声明的增量链：`generic capability/headroom -> failure-surface geometry -> patch opportunity -> typed conditional response`。只有最后一层明显提供增量时，才支持完整 TRACE-H 方法主张；Metric Freedom 或前一层若已足够，则必须删除更复杂主张。

### 关键消融

- 去掉 baseline success 状态；
- 去掉 trigger depth/budget bin；
- 不做来源模型平滑；
- 只用 activation count，不用 first-trigger type；
- 把三个 patch 合并成一个通用条件效应；
- 不使用 abstention；
- 用 always-on prose guidance 替代 event-gated patch，作为契约外负对照而非主贡献。

## 11. 指标与统计分析

### 11.1 效应预测

- task-level expected-delta MAE；
- category-patch aggregate effect MAE；
- 以 `|Delta_Y| >= 2/12` 为主要实际效应阈值、`1/12` 为敏感性阈值的符号准确率；
- rescue、harm 和 unchanged 三类概率的 Brier score 与校准；
- 50%、80%、95% bootstrap interval coverage，仅作经验评价。
- 三个 patch 的 pairwise ranking accuracy 与 top-1 patch/none accuracy；
- 在相同 Metric Freedom bin 内的 patch-specific discrimination，防止结果只来自 generic headroom。

### 11.2 部署评价

- negative-transfer rate；
- 基于 source LOMO 与 target 共 36 个 held-out category-patch cells 的 risk-coverage curve；真正 prospective target 的 9 个点单独标示并按离散 coverage 报告；
- 相对 full oracle 的 patch-choice regret；
- 每避免一次负迁移所放弃的正收益；
- choose none、choose P1/P2/P3、reject 与 abstain 的数量和类别分布；
- 与 active-target probe oracle 的成本-遗憾差距。

### 11.3 机制评价

- 每个模型/patch 的 opportunity rate；
- `P(rescue | trigger)` 与 `P(harm | trigger)`；
- first-trigger depth 与剩余预算；
- raw effect 与机会标准化效应的模型间异质性；
- patch 触发后新增 step、token、wall time 和失败模式。

### 11.4 推断纪律

- 任务为主要重采样单位，不能把 step 当作独立样本；
- 对模型数量只有四个保持明确，不用渐近正态理论制造虚假精度；
- paired binary success 使用配对差和 exact/bootstrap interval；
- category 和 patch 均为预声明固定效应；
- 所有阈值、回退、排除和基础设施失败规则在 seal 前冻结；
- 主结论不依赖 LLM judge。

## 12. 成功标准、转向标准与停止标准

### 12.1 72 小时继续条件

必须同时满足：

1. 至少两个 patch 在三个 pilot 模型中的至少两个模型上各触发至少四个任务；
2. no-trigger invariance 通过；
3. 至少一个 patch 出现跨模型 rescue/harm 差异或总体 effect disagreement；
4. Metric Freedom diagnostic 能按预声明协议稳定计算，且不会读取 patch outcomes；
5. 在一个预先指定的 sealed pseudo-target 上，TRACE-H 至少在一个主要 patch-specific 指标上明确优于 `MF + source prior`、baseline strength 与 source aggregate copy；
6. 该增量能判断 H-MF/H-FS/H-OPP/H-RESP 哪一层值得进入全矩阵，不能只在事后 case 上成立；
7. 约 936 episodes 的全矩阵运行时间、成本和 telemetry 可承受。

### 12.2 Pivot

- `pi` 强烈变化但 `rho` 近乎常数：转为更简洁的 failure-opportunity standardization paper。
- Metric Freedom 与 baseline strength 已解释 generic gain，但无法区分 patch：保留 patch-choice 问题，删除无增量的 generic predictor 叙事。
- Metric Freedom-informed baseline 与 TRACE-H 持平：转为 Metric Freedom 独立复核/边界研究；当前 AAAI 主方法停止。
- failure-surface-nearest 或 opportunity-only 与 TRACE-H 持平、但共同胜过 capability/source-copy：转为 H-FS/H-OPP 的 prospective sealed validation，删除不必要的细分响应模型。
- `rho` 变化但可由 trigger severity 解释：保留 TRACE-H，缩小到一个 patch family。
- 只有 verifier gate 有信号：转为 verifier tax 的 prospective prediction，直接对比 The Verifier Tax。
- 只有完整 harness 差异、原子 patch 无效：放弃本主张，不能退化成 Harness-Bench 复现。

### 12.3 Stop

- patch 很少触发或 no-trigger 条件仍改变行为；
- repeated baseline 无法稳定计算 Metric Freedom，或 MF protocol 只能用事后调参实现；
- 条件效应比 raw effect 更不稳定；
- TRACE-H 与 `MF + source prior`、baseline strength 或更简单 target-aware 方法持平/更差，且没有独立 patch-specific sealed decision 发现；
- 目标 sealed audit 显著失败，且不能诚实改写成独立的负结果贡献；
- 发现已有论文完成同一 prospective trigger-standardized target-effect audit；
- 2026-07-16 前 reference harness 与四个 endpoint 仍不能稳定运行。

## 13. AAAI 审稿维度对齐

| AAAI 维度 | 旧版风险 | TRACE-H 改进 | 仍需的证据 |
|---|---|---|---|
| Significance | 像 harness 调参工程 | 直接回答模型升级时能否安全保留 runtime patch | 至少一次真实负迁移被提前识别 |
| Novelty | Metric Freedom 已做 baseline-only a priori utility prediction；SEAGym 已提出事后 H-FS；多篇工作已做 frozen transfer | 控制 generic headroom 后，对多个 atomic patches 做 patch-specific signed/calibrated sealed choice | 必须胜过 MF family、baseline strength、opportunity-only 与 failure-surface-nearest；不能只靠 conjunction claim |
| Soundness | 高维小样本回归 | dormant contract、配对效应、低维标准化、明确假设 | contract tests 与 raw traces |
| Empirical quality | 多轴稀疏矩阵 | 36 任务、4 模型、3 原子 patch、MF repeats、LOMO + sealed target | 完整约 936 episodes |
| Clarity | 多个方法中心 | 一张图、一个分解式、一个部署曲线 | 七页正文严格收缩 |
| Relevance | harness 术语偏工程 | 连接 agent reliability、OOD generalization、selective prediction | 真实可执行任务而非 toy only |
| Reproducibility | API 漂移 | 固定离线 tasks、patch contracts、manifest hash、append-only traces | 开源 runner 与一键分析 |

[AAAI-27 Main Technical Track](https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/) 当前官方标准强调贡献的 significance、novelty、theoretical/empirical soundness、relevance、clarity、responsible research 和 reproducibility，并明确主内容上限为七页；本表不是写作装饰，而是实验取舍依据。

## 14. 十七天执行计划

[AAAI-27 官方时间表](https://aaai.org/conference/aaai/aaai-27/) 为 2026-07-21 摘要截止、2026-07-28 正文截止、2026-07-31 补充材料与代码截止，均为 UTC-12 23:59。

| 日期 | 必须完成 | 决策门 |
|---|---|---|
| 7/11 | idea、contracts、任务选择规则、本文档冻结 | 不再扩张主问题 |
| 7/12 | reference harness、episode schema、P1-P3 单测、MF diagnostic | no-trigger invariance 与 `F_out` 可自动测 |
| 7/13-7/14 | 12 任务 x 3 模型、约 234 episodes pilot | 必须胜 MF-informed pseudo-target baseline，否则 pivot/stop |
| 7/15-7/17 | 36 任务 x 3 source 全矩阵与 baseline repeats | telemetry 完整、成本可控 |
| 7/18-7/19 | LOMO、MF family、ablation、trigger taxonomy 冻结 | TRACE-H 必须胜最强 generic/simple baseline |
| 7/20 | 目标 baseline、prediction file 与 seal | 禁止再看 target patch outcome |
| 7/21 | 摘要提交；只陈述已得到的结果 | 不预支未完成结果 |
| 7/22-7/23 | target patch audit 与首份 score report | 成功则继续主会稿，失败则诚实转向 |
| 7/24-7/25 | 统计、图表、机制与 robustness | 三张 killer figure 完成 |
| 7/26-7/27 | 七页正文、related work refresh、claim audit | 导师完整审阅 |
| 7/28 | 正文提交 | 无未验证核心数字 |
| 7/29-7/31 | 代码、数据、补充材料 | 可从干净环境复现 |

## 15. 工作量与资源

| 工作包 | 聚焦人时估计 | AI assistance 可承担 | 不可外包风险 |
|---|---:|---|---|
| reference harness 与 telemetry | 14-24 | 代码骨架、测试、schema | endpoint/CLI 调试 |
| 三个 patch 与 contract tests | 10-16 | 实现和边界用例 | 原子性判断 |
| pilot、MF repeats 与 source matrix | 16-26 监督 | 日志归档、距离矩阵、失败聚类 | API 延迟、额度、异常 |
| TRACE-H 与 MF/simple baselines | 14-22 | 分析代码、bootstrap | 公平信息预算与泄漏审计 |
| target seal 与 audit | 6-10 | hash、自动评分 | 严格执行协议 |
| 图表、正文、附录 | 24-36 | 初稿、压缩、格式 | 贡献叙事与事实核查 |
| 最终检索与复现 | 6-10 | 搜索和清单 | 新碰撞判断 |

合计约 **100-155 个聚焦人时**，另加模型执行墙钟时间。重度 AI assistance 能明显压缩编码、日志处理和写作，但不能代替 endpoint 稳定性、实验完整性、科学判断和最终作者责任。

当前不把 ToolBench-X 作为主环境：截至 2026-07-11，其公开仓库 commit `cf64912d0606808e706618f59ef54ec6a1c12712` 只有“full release soon”的 README，论文所称 code/data 尚不可复现。它可作为相关工作或代码真正发布后的附加验证，不能成为 deadline-critical dependency。

## 16. 预期图表

### Figure 1：TRACE-H 一图讲清

来源 paired outcomes 形成 conditional rescue profile；目标 baseline traces 形成 generic headroom 与 patch-specific failure surface；二者结合得到 sealed effects 和 patch/none/abstain choice。

### Figure 2：预测效应对真实效应

横轴为预测 `Delta`，纵轴为 unsealed `Delta`；灰色为 source LOMO，红色边框为真正 prospective target；并列 source mean、baseline strength、MF-headroom、MF-nearest、Source x MF、failure-surface-nearest、opportunity-only 与 TRACE-H。

### Figure 3：风险-覆盖与部署遗憾

展示 abstention 从 0 到高覆盖时 negative-transfer rate 和 patch-choice regret；比较 always reuse、source-copy、MF family、capability-nearest、failure-surface-nearest、opportunity-only 与 TRACE-H。

### Table 1：Patch contract

列出 trigger、observable input、intervention、最大预算、禁止信息、no-trigger test 和可能伤害机制。

### Table 2：救援稳定性

列出 raw effect heterogeneity、trigger-conditioned rescue/harm、LOMO error 和 prospective target error。

## 17. 允许与禁止的论文主张

### 若结果成功，可主张

1. event-gated harness patch 的总体效应可被 baseline failure opportunity 与条件 response 分解；
2. 在本实验中，逐层比较 Metric Freedom/generic headroom、surface geometry、opportunity 与 conditional response 后，说明哪一层足以区分 held-out model 上的多个 patch；只有 TRACE-H 明显胜出时才主张 typed conditional response 的增量；
3. baseline-only target traces 能支持 prospective patch/none/abstain choice，并减少本任务集上的负迁移与 choice regret；
4. 完整 traces、contracts 和 seal 提供一个可复查的 model replacement audit。

### 即使结果成功，也禁止主张

- 所有 harness effect 都可迁移；
- 发现了普适 causal transport law；
- 提供 distribution-free risk guarantee；
- 对任何新模型都可零成本判断；
- 三个 patch 代表全部 harness 设计；
- 36 个任务支持基础模型能力的一般排序；
- 事件触发 patch、可执行 skill 或 harness benchmark 是本文首创。
- baseline-only a priori skill-utility prediction 是本文首创；
- Metric Freedom 只能作为弱相关工作而非第一强基线。

## 18. 七页论文结构

1. **Introduction，0.8 页：** 模型替换问题、矛盾证据、一句话贡献。
2. **Related Work，0.7 页：** Metric Freedom、complete harness transfer、selectors、effect prediction、selective prediction。
3. **Problem and Contract，1.0 页：** dormant patch、estimand、seal。
4. **TRACE-H，1.2 页：** 分解、估计、uncertainty、abstention。
5. **Experimental Protocol，0.9 页：** tasks、models、patches、baselines。
6. **Results，1.7 页：** heterogeneity、LOMO、sealed target、regret。
7. **Analysis and Limitations，0.5 页：** rescue stability、failure cases、边界。
8. **Conclusion，0.2 页。**

正文只保留一个方法中心。广泛 taxonomy、第二 benchmark、复杂 Bayesian variant、所有 case study 和额外成本曲线放补充材料。

## 19. 四年研究延展性

若 TRACE-H 的机制成立，四年主线可以自然扩展为：

1. 从单一 trigger patch 扩展到组合 patch 的 interference-aware transport；
2. 从 baseline-only traces 扩展到主动选择最有信息量的 target probes；
3. 从固定 model replacement 扩展到在线 version drift monitoring；
4. 从经验 interval 扩展到假设明确、样本充分的 selective-risk guarantees；
5. 从任务成功扩展到安全、成本与权限约束下的多目标部署。

若第一步失败，也能得到明确结论：failure type 仍不足以稳定条件效应，后续研究应转向少量 target intervention probes，而不是继续追求纯 baseline-only transfer。

## 20. 当前决策

**保留 TRACE-H，但从“唯一推荐主线”降为 collision-aware conditional go，只进入包含 Metric Freedom 的 72 小时杀伤实验。**

它相较旧版仍更可证伪、参数更少，但 Metric Freedom 已占据 broad baseline-only predictor framing。SEAGym 是 H-FS 假说来源，LIFE-HARNESS/HarnessFix 是广泛迁移先验，Metric Freedom 是 generic headroom 第一强基线；三者都必须正面承认。本文能否形成完整方法贡献，取决于 sealed patch-specific prediction 是否明显胜过 MF family、baseline strength、opportunity-only 与 failure-surface-nearest。若 Metric Freedom 已足够，当前主方法应停止；若简单 opportunity 层足够，则主动简化；只有 typed response 再提供增量时才保留完整 TRACE-H。项目当前没有自己的 pilot、source matrix 或 sealed-target 结果，idea 总评与接受概率以 [DR-0003](../decisions/0003-metric-freedom-collision-reassessment-zh.md) 为准。
