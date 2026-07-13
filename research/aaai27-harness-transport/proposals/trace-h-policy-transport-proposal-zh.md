# TRACE-H 正式 Proposal：向未见 LLM Executor 运输可执行 Harness 控制策略

- **英文工作题目：** TRACE-H: Transporting Harness Control Policies to Unseen LLM Executors
- **TRACE-H：** Trajectory-Response Alignment for Cross-Executor Harnesses
- **目标会议：** AAAI-27 主会
- **版本日期：** 2026-07-11，design-method expansion 版
- **论文类型：** 新方法 + end-to-end agent system evaluation，不是诊断/测量论文
- **状态：** 先做 72 小时方法杀伤实验；必须在目标任务效用上击败最强同预算 baseline
- **取代：** [上一版效应预测 Proposal](trace-h-formal-proposal-zh.md)的 method center；旧文保留作证据边界和协议历史
- **术语导读：** 不熟悉 agent/RL/OT 的读者先看[导师版术语与符号说明](../notes/trace-h-terminology-guide-zh.md)

## 摘要

基础模型升级时，旧 agent harness 往往被整体照搬、重新搜索或在目标任务上反复试错。现有方法已经能为已知 backbone 搜索和改写 skills、把 skill 编译到不同 model-harness target，或在固定 executor 的离线轨迹上学习 harness controller；但它们尚未解决一个更直接的部署问题：**能否在不收集任何目标 harness-action outcome 的条件下，把来源模型上学到的运行时控制策略运输到一个未见 LLM executor？**

TRACE-H 提出 Branch-and-Transport Harness Policy。首先，在来源 executor 的同一 trajectory prefix 上分支执行 `NONE/CHECK/RETRY/REPLAN` 等结构化 harness actions，直接获得事件级 action advantage，建立反事实 response bank。其次，只运行目标 executor 的无干预 baseline trajectories，用 event type、结构化执行状态和局部语义构造 transport cost，通过 partial unbalanced optimal transport 将目标 states 对齐到具有相似 action-response 的来源 states；无法可靠匹配的目标状态保留为 unmatched mass，而不是被强行迁移。最后，TRACE-H 将运输后的 action lower confidence bound 编译为轻量 runtime router，在每个触发状态选择最有希望的 action 或 `NONE`，直接产生可执行 target harness policy。

论文的主结果不是预测相关性，而是未见 executor 上的任务成功率、成本调整效用、负干预率和相对 target oracle 的 policy regret。实验以 ALFWorld 为直接方法 PK 主场，并在 WebShop 做第二环境验证；对 Qwen3 同家族 held-out executor 和 Gemma 跨家族 executor 完全封存 intervention outcomes。TRACE-H 必须正面比较 Offline-RL Harness、MASA、SkillAdaptor、SkVM-style capability adaptation、Metric-Freedom gating、source-only controller、nearest-executor、kNN response transport 和 balanced OT。只有在零目标干预预算下显著超过最强同预算方法，且在成本效用前沿上接近使用目标反馈的方法，才支持“跨 executor harness policy transport”这一方法贡献。

## 0. 先把术语讲清楚

### 0.1 系统由什么组成

```text
Agent system
  = executor          真正生成下一步行为的基础模型
  + harness           模型外部的执行控制逻辑
  + tools/environment 行为实际作用的工具与任务环境

Harness policy
  = 当前执行状态 state -> 选择 NONE/CHECK/RETRY/REPLAN
```

本文不修改 executor 权重。TRACE-H 学习的是模型外部的 harness policy。

### 0.2 核心对象

| 术语 | 本文中的严格含义 |
|---|---|
| **Backbone / 基础模型** | 提供语言理解和生成能力的 LLM checkpoint，如 Qwen3-8B |
| **Executor / 执行模型** | 在 agent loop 中接收当前上下文并实际产生下一步 action 的模型实例 |
| **Harness** | 位于模型权重之外，管理观察、工具、检查、重试、重规划和终止的运行层；不只是一段 prompt |
| **Harness action** | Harness 对执行过程采取的控制操作；不是环境动作 |
| **Harness policy** | 从当前 state 到 harness action 的决策规则 `pi(a|s)` |
| **Runtime router** | 在运行中读取 state、调用 policy 并执行 action 的轻量组件 |
| **Source executor** | 允许收集不同 harness-action outcomes、用于学习 response 的模型 |
| **Target executor** | 希望部署新 policy 的模型；freeze 前禁止读取其非 NONE action outcomes |
| **Unseen executor** | 整个模型的 action outcomes 在训练中留出；仍允许公开 metadata 和 baseline trajectories，并非完全零访问 |
| **Whole-executor holdout** | 留出一个完整基础模型，而不只是留出若干 tasks |

### 0.3 什么叫“零目标干预”

| 术语 | 定义 |
|---|---|
| **Baseline / NONE** | 不加入 CHECK、RETRY、REPLAN，让 agent 按固定参考 harness 运行 |
| **Baseline trajectory** | NONE 条件下从任务开始到结束的 observations、actions、tool results 与 final outcome |
| **Target baseline calibration** | 用目标 baseline trajectories 了解它会遇到哪些 states，但不测试任何非 NONE action |
| **Target action feedback** | 在目标模型上实际运行 CHECK/RETRY/REPLAN 后得到的 success、reward、cost 或失败信息 |
| **Target-intervention-free** | policy freeze 前 target action feedback 为零；不等于 target calls 为零 |

### 0.4 Trajectory、state、event 与 branch

| 术语 | 定义 |
|---|---|
| **Trajectory** | 一次任务从开始到终止的执行序列 `s0,a0,s1,a1,...` |
| **State** | Router 在一个时刻被允许读取的信息摘要，如进度、错误、工具结果和剩余预算 |
| **Event / trigger** | 允许考虑非 NONE action 的预声明时刻，如 invalid action 或 no progress |
| **Trajectory prefix** | 从任务开始到某个 event 为止的历史 |
| **Snapshot/replay** | 保存环境状态，或按原动作重放到相同 state；state hash 用来确认分支起点一致 |
| **Branch** | 从同一 prefix 复制多个起点，分别实际运行不同 harness actions |
| **Counterfactual branch** | 对“当时若改用另一 action 会怎样”的实验性分支；不是让 LLM 口头猜测 |
| **Action advantage** | `A_i(a)=G_i(a)-G_i(NONE)`；表示 action `a` 相对不干预改善或伤害多少最终效用 |
| **Counterfactual Branch Bank** | 全部 source states、分支 outcomes、advantages、成本与方差组成的 response 数据库 |

### 0.5 四个 actions

| Action | 含义 | 边界 |
|---|---|---|
| `NONE` | 不增加控制，按 baseline 继续 | 不允许偷偷加入额外提示或调用 |
| `CHECK` | 检查当前动作所需的状态、证据或前置条件 | 不直接提供任务答案 |
| `RETRY` | 对刚失败或无效的 action 做一次规范化重试 | 不对正常 action 无限重试 |
| `REPLAN` | 根据当前状态生成最多三步的短计划 | 不让外部 planner 接管整个任务 |

### 0.6 “Transport”不是复制 prompt

**Policy transport** 指把 source states 上观察到的 `action -> terminal outcome` 规律，用来构造 target 的 `state -> action` policy。运输对象是控制响应，不是模型权重或一段静态文本。

| 术语 | 定义 |
|---|---|
| **State representation `z`** | 把复杂 prefix 转为结构特征与局部文本 embedding 组成的可比较向量 |
| **Transport cost `C_ij`** | source state `i` 与 target state `j` 的不相似度 |
| **Optimal Transport / OT** | 在两个 state 分布之间寻找总体匹配代价最小的软对应关系 |
| **Transport plan `Gamma`** | target state 从各 source states 借多少 response 证据的权重矩阵 |
| **Mass** | state 在匹配问题中的统计权重，不是内存大小 |
| **Balanced OT** | 强制全部 source/target mass 都匹配，即使部分 states 并不相似 |
| **Partial/Unbalanced OT** | 允许只匹配有可靠支持的部分 states，不可靠部分可留空或付惩罚 |
| **Unmatched mass** | target state 没有可信 source 对应时保留的未匹配权重 |
| **Target-private state** | 目标模型独有、source bank 中没有可靠对应的 state；private 不是隐私含义 |
| **Response-aware** | metric/hyperparameters 按 source action-response utility 选择，而不是只按文本相似度 |

### 0.7 如何从估计变成动作

| 术语 | 定义 |
|---|---|
| **Transported action advantage** | 按 `Gamma` 加权 source advantages，估计 target state 上各 action 的收益 |
| **LCB** | Lower Confidence Bound，下置信界；把估计收益的不确定性扣掉后的保守下端 |
| **Support coverage** | target state 获得多少可靠 source response 支持 |
| **Conservative policy** | 只有 action 的 LCB 扣除成本后仍为正且 coverage 足够才干预，否则 NONE |
| **Policy compilation** | 把冻结的 features、weights、thresholds 和 fallback 序列化为 router artifact；不是编译机器代码 |
| **Policy distillation** | 用小型树模型/MLP 模仿 OT router 以降低开销；不读取 target action rewards |

### 0.8 论文到底用什么判胜负

| 术语 | 定义 |
|---|---|
| **End-to-end** | 从任务输入到最终环境结果完整执行，而不是只评价推荐或预测 |
| **Normalized utility** | success 收益减去预声明的 tokens、steps 和 invalid-action 成本；必须同时报告原始 success |
| **Negative intervention** | 所选非 NONE action 的真实 return 低于同 state 的 NONE |
| **Oracle** | 事后知道 target 各 action 真值、总能选最优 action 的评估上界；部署时不可用 |
| **Policy regret** | Oracle return 减去实际 policy return；越小越好 |
| **Same-budget baseline** | source data、target baseline、target action feedback 和测试成本均与 TRACE-H 对齐的方法 |
| **Primary metric** | 决定论文成败的 success、utility 和 regret |
| **Secondary metric** | 解释机制的 effect MAE、OT cost、coverage 和 calibration；不能单独支撑论文 |

### 0.9 常用缩写

| 缩写 | 全称与含义 |
|---|---|
| `MDP` | Markov Decision Process；用 state、action、transition、reward 和 horizon 描述顺序决策问题 |
| `RL` | Reinforcement Learning；根据执行回报学习 policy |
| `AW` | Advantage-Weighted Regression；让高 advantage 的 action 样本获得更大模仿权重 |
| `MF` | Metric Freedom；从 repeated baseline outputs 估计 generic skill headroom 的已有指标 |
| `OT` | Optimal Transport；在两个 state 分布之间求低代价软匹配 |
| `LCB` | Lower Confidence Bound；估计收益的保守下界 |
| `kNN` | k-Nearest Neighbors；直接参考最相似的 `k` 个 source states |
| `MAE` | Mean Absolute Error；预测值与真值绝对误差的平均 |
| `CI` | Confidence Interval；这里主要指 bootstrap 得到的差异不确定区间 |
| `KL` | Kullback-Leibler divergence；unbalanced OT 中惩罚 transport marginals 偏离原分布的项 |
| `MLP` | Multi-Layer Perceptron；可用于蒸馏 router 的小型前馈网络 |
| `LOMO/LOEO` | Leave-One-Model/Executor-Out；轮流留出整个 source executor 做伪目标验证 |
| `AOT/JIT` | Ahead-of-Time / Just-in-Time；SkVM 在执行前编译与运行中重编译的两种阶段 |

一个从 invalid action 到 partial transport 再到 CHECK 决策的完整 ALFWorld 例子，以及 AW、MASA、SkVM、SkillAdaptor、MF、kNN、PAR、AOT/JIT 和全部数学符号，见[完整术语表](../notes/trace-h-terminology-guide-zh.md)。

## 1. 论文中心

### 1.1 大问题

Harness 不只是提示词集合，而是作用于 agent trajectory 的外部控制策略。基础模型变化相当于执行器 dynamics 变化：同一个 `CHECK`、`RETRY` 或 `REPLAN` action 的后果会改变。真正需要运输的不是一段文本，而是：

```text
在什么执行状态下，采取哪个 harness action，能以多大概率改善最终任务效用？
```

### 1.2 一句话方法

> TRACE-H 在来源模型的同一轨迹前缀上分支测量 harness-action response，再把目标模型的无干预 states 部分对齐到 response bank，编译出一个对无支持状态自动选择 NONE 的目标 runtime controller。

### 1.3 非诊断性硬约束

论文的第一主表必须报告部署后的 end-to-end utility。以下结果单独出现时均视为失败：

- 只预测哪个 patch 有益；
- 只报告 effect MAE、相关系数或 calibration；
- 只画 model-harness interaction heatmap；
- 只证明 failure states 可聚类或 OT 距离有意义；
- router 改变了 CHECK/RETRY 频率，但没有提高最终任务质量；
- 只胜 `NONE` 或固定规则，没有胜 learned controller 与 model-aware adaptation。

## 2. 预期贡献

1. **新任务：Cross-Executor Harness Policy Transport。** 来源侧允许 harness-action outcomes，目标适配侧只允许 baseline trajectories；输出是可执行 state-conditioned policy，而非评分报告。
2. **新数据机制：Counterfactual Branch Bank。** 在相同 executor、task 和 trajectory prefix 上分支不同 harness actions，获得比整条轨迹相关性更干净的 event-level action advantage。
3. **新运输机制：Response-Aware Partial Transport。** 对齐对象不是文本相似度，而是与 action response 有关的执行状态；partial OT 允许 target-private states 不迁移。
4. **新决策机制：Conservative Policy Compilation。** 将 action value、transport uncertainty、action cost 和 support coverage 编译成 `action/NONE` router，直接优化目标任务效用。
5. **理论目标：Transport-Regret Bound。** 把 target policy regret 分解为 source response estimation、state alignment cost、unmatched mass 和 finite-sample uncertainty。
6. **实证目标：与现有设计方法正面 PK。** 在共同模型、任务和预算上比较，而不是只用 related-work 文字区别。

## 3. 扩张后的证据边界

| 近邻方法 | 已经占据的贡献 | TRACE-H 必须新增并击败的轴 |
|---|---|---|
| Offline-RL Harness, 2607.05458 | 固定 executor 上把 harness 建模为 MDP，用 AW 学 state-conditioned controller | 整个 target executor 留出；不读取 target action rewards；运输 source policy |
| MASA, 2605.30723 | target model card + target feedback 搜索/改写 skills；rewriter 做 unseen task/environment | unseen backbone/family；runtime action policy；零 target intervention feedback |
| SkVM, 2604.03088 | 对 model-harness target 做 capability profiling、AOT/JIT skill compilation | 运输 event-level action response；动态 policy；不等 target failures 后再 JIT |
| Adaptive Auto-Harness, 2606.01770 | harness tree、task-category solve-time routing、stream feedback | 未见 executor 上的 state-level routing；无 target outcome adaptation |
| SkillAdaptor, 2606.01311 | target failure localization、skill modification、target rerun qualification | 不读取 target failed patch trajectory；不重跑候选修改即可部署 |
| SkillSelect-Serve, 2607.00011 | 预算/QoS 约束的 skill bundle recommendation | actual execution utility 为主指标；action response 而非 curated relevance/utility |
| Bayesian-Agent, 2606.08348 | 同 backend 在线 posterior evidence 与 skill rewrite policy | source-to-unseen-executor transport；目标零 action evidence |
| Metric Freedom, 2604.01608 | baseline-only generic skill headroom | 同一 headroom 下的 state/action-specific executable policy |
| Partial Harnessing, 2605.21516 | harness coverage 的 alignment theory 与静态 stopping principle | 从来源 action outcomes 学动态、跨 executor 的状态条件控制 |

因此，论文不得声称首个 learned harness、首个 model-aware skill、首个 skill compiler、首个动态 router 或首个 target baseline signal。方法主张只落在：

> target-intervention-free、event-level、state-conditioned、cross-executor executable policy transport。

详细证据见[方法扩张全文审计目录](../foundations/method-wave/README.md)与[MASA 全文证据卡](../foundations/method-wave/notes/2605.30723-masa-fulltext-evidence-zh.md)。

## 4. 问题定义

设 `m` 为 LLM executor，环境任务为有限时域 Harness MDP：

```text
M_m = (S, A_H, P_m, R, H)
```

- `S`：可观测执行状态，包括进度、最近 observation/action、错误类型、验证状态、剩余预算和局部文本；
- `A_H = {NONE, CHECK, RETRY, REPLAN}`：固定、跨模型一致的结构化 harness actions；
- `P_m`：由 executor `m` 与环境共同决定的 transition dynamics；
- `R`：最终任务成功与成本组成的效用；
- `H`：最大执行步数。

来源 executor 集合为 `M_S={m1,...,mK}`，允许收集不同 harness-action outcomes。目标 executor `mT` 在适配阶段只允许产生 `A_H=NONE` 的 baseline trajectories。任务是学习：

```text
pi_T(a | s), a in A_H
```

并在从未用于目标 action adaptation 的 held-out tasks 上最大化：

```text
J_T(pi) = E[success - lambda_token * token_cost
                     - lambda_step * extra_steps
                     - lambda_fail * invalid_actions]
```

主分析同时报告原始 success，避免效用权重掩盖任务失败。

## 5. TRACE-H 方法

### 5.1 Stage A：Counterfactual Branch Bank

在每个 source baseline trajectory 中识别预声明 event：

- invalid or rejected action；
- repeated/no-progress state；
- evidence or precondition missing；
- pre-submit uncertainty；
- budget pressure。

对事件前缀 `x=(s_0,a_0,...,s_t)` 保存环境 snapshot；若环境不支持 snapshot，则确定性 replay 到同一状态并验证 state hash。随后从同一前缀分别执行所有合法 harness actions并滚动到终局：

```text
G_i(a) = terminal utility after branching action a at prefix i
A_i(a) = G_i(a) - G_i(NONE)
```

每个 branch point 保存 `(z_i, action mask, A_i, variance, cost)`。这里的 `A_i(a)` 是 method supervision，不是论文终点。

相较于普通 offline trajectories，branch bank 有两个作用：

1. 同一 prefix 上覆盖多个 actions，降低行为策略 support 缺口；
2. 直接区分“这个状态需要干预”与“某类高分轨迹经常包含干预”。

### 5.2 Stage B：Transport State

每个事件状态编码为：

```text
z = [event_type,
     progress_and_budget,
     verifier_and_tool_flags,
     recent_action_outcome,
     compact_text_embedding]
```

结构字段使用确定性 extractor；文本 embedding 只覆盖最近 observation/error 与当前 subgoal，不输入完整 chain-of-thought。所有 scaler、embedding model、字段权重在 source leave-one-executor-out 中冻结。

### 5.3 Stage C：Response-Aware Partial Optimal Transport

目标 baseline calibration trajectories 产生未标注 states `{z_j^T}`。来源 branch bank 为 `{z_i^S,A_i}`。transport cost 为：

```text
C_ij = alpha * structural_distance(z_i^S, z_j^T)
     + beta  * semantic_distance(z_i^S, z_j^T)
     + infinity * I[event_type incompatible]
```

求 partial/unbalanced transport plan：

```text
Gamma* = argmin_Gamma <Gamma,C>
         + eps * entropy(Gamma)
         + tau_s * KL(Gamma 1 || p_S)
         + tau_t * KL(Gamma^T 1 || p_T)
```

关键设计不是“使用 OT”本身，而是让目标状态可保留 unmatched mass。强行 balanced alignment 会把 target-private failure state 映射到错误来源 response，造成负迁移。

### 5.4 Stage D：Transported Action Advantage

对目标状态 `j` 与 action `a`：

```text
Ahat_T(j,a) = sum_i Gamma_ij * A_i(a) / sum_i Gamma_ij
```

不确定性由三部分组成：source branch variance、跨 source executor disagreement、transport concentration/coverage。用 source executor bootstrap 得到 `LCB_T(j,a)`。

为使 transport 对 action response 而非只对表面语义敏感，source-side metric weights 通过 leave-one-executor-out policy utility 选择；不能读取 target action outcomes。

### 5.5 Stage E：Conservative Policy Compilation

运行时 policy 为：

```text
score(j,a) = LCB_T(j,a) - lambda_cost * action_cost(a)

pi_T(j) = argmax_a score(j,a),  if max score > 0 and coverage >= kappa
          NONE,                  otherwise
```

输出不是自然语言建议，而是版本化 router artifact：

- feature schema/hash；
- event/action masks；
- source branch-bank hash；
- transport hyperparameters；
- compact target prototypes；
- deterministic action decision；
- fallback `NONE`。

第一版每个 episode 最多执行一次非 `NONE` action，避免未经目标数据验证的 intervention-induced state distribution compounding。扩展实验再允许多次控制。

### 5.6 Stage F：Router Distillation

partial OT 可离线生成 target pseudo-label 与 margins。为降低运行时开销，将决策蒸馏为小型 gradient-boosted tree 或两层 MLP；蒸馏模型必须在 target baseline states 上复现原 policy 的 action/none 决策，不能用目标 action reward训练。主结果同时报告原始 OT router 与 distilled router。

### 5.7 理论目标

假设 action advantage 在 transport metric 下为 `L`-Lipschitz，source branch estimate error 不超过 `eps_src`，matched target mass 为 `1-rho`，平均 transport cost 为 `W_partial`。目标证明形式为：

```text
J_T(pi*_T) - J_T(pi_TRACE)
  <= 2 * (eps_src + L * W_partial + eps_finite)
     + rho * R_range
```

该 bound 直接对应四个可操作改进方向：更多 source branch repeats、学习更好的 state metric、收紧 finite-sample interval、对 unmatched state 选择 `NONE`。若无法给出严格 theorem，正文至少给 proposition 与完整假设，不把经验 OT 包装成理论保证。

## 6. 为什么它是设计方法

TRACE-H 改变 target agent 的执行过程并直接产生更高效用：

```text
baseline trajectories
        -> target state support
source branch responses
        -> transported action values
        -> executable runtime router
        -> higher target task utility
```

预测误差只回答机制是否按预期工作；论文成败由最后一箭决定。主标题、摘要、主表和结论均不得以 diagnosis 为中心。

## 7. 实验对象

### 7.1 环境

1. **ALFWorld：** 主战场。可复用 MASA 开源 runner/skills，环境可 replay，适合事件级 branch。
2. **WebShop：** 第二环境。验证运输规律不只适用于 embodied-text action grammar。

若 WebShop 在 2026-07-16 前无法稳定 branch/replay，则只保留一个预声明小规模 external validation，不以不完整第二环境拖垮主实验。

### 7.2 Executor split

主 split：

- source：Qwen3-4B、Qwen3-8B、Qwen3-14B；
- held-out in-family target：Qwen3-32B；
- held-out cross-family target：Gemma3-12B 或在 24GB 显存上稳定的同级开源模型；选择规则在任何 target action run 前冻结。

所有模型使用相同 environment wrapper、action parser、temperature 和最大步数。不能用 target action outcomes 挑量化版本或 prompt。

### 7.3 数据 split

- source branch-train tasks；
- source LOMO validation tasks；
- target baseline-only calibration tasks；
- disjoint target final-test tasks。

目标 calibration split 只运行 `NONE`。所有 target method actions 只在 final-test 阶段第一次执行。

## 8. 方法 PK 矩阵

### 8.1 第一主表：相同目标信息预算

所有方法共享 source data、target baseline calibration trajectories、model metadata 与 inference budget。

| 方法 | 作用 | 必跑 |
|---|---|---|
| No Harness | 原始 executor | 是 |
| Always CHECK/RETRY/REPLAN | 固定规则 | 是 |
| Source-Best Static | 来源平均最优单 action | 是 |
| Source-AW | 用 Offline-RL Harness 官方代码思路在 pooled source 上训练，原样迁移 | 是 |
| Nearest-Executor AW | 用 target baseline fingerprint 选一个 source controller | 是 |
| MF-Gated AW | Metric Freedom 只决定是否启用 source controller | 是 |
| kNN-Branch | 最近 source branch 的 action advantage | 是 |
| Balanced-OT | 不允许 unmatched mass 的 OT transport | 是 |
| TRACE-H | branch bank + partial OT + LCB/NONE | 是 |

TRACE-H 必须胜这一表中的最强方法，而不是只胜 No Harness。

### 8.2 第二主表：直接论文方法

| 方法 | 目标反馈预算 | 复现方式 | 比较角色 |
|---|---:|---|---|
| MASA Base Skill | 0 | 官方 JSON/runner | model-agnostic skill baseline |
| MASA DS-Adapter | 0 action rewards | 官方 artifact；必要时按论文做一次 model-card rewrite | model-aware static rewrite |
| MASA evolved skills | 数百至数千 target rollouts | 官方已发布 Qwen skill JSON | target-feedback upper comparator |
| SkillAdaptor | 多轮 target failures + qualification reruns | 官方代码 snapshot | active target adaptation |
| Offline-RL Harness target-trained | target rollout buffer | 官方代码 snapshot | target-policy oracle comparator |
| SkVM-style AOT | target capability probes；JIT 另计 target failures | 忠实重实现可支持的 capability compensation 子集并明确非官方 | capability compiler comparator |
| Partial Harnessing | 无 target action reward或少量 profile | 按论文 stopping principle 实现 | theory-driven static policy |

由于不同论文 action space 不完全一致，必须同时提供：

1. **system-level comparison：** 各方法按自己的完整公开流程运行，比较 end-to-end success/cost；
2. **controlled action-space comparison：** 在统一 `A_H` 上比较 transport/controller 机制；
3. **target information ledger：** 报告每种方法读取多少 target action outcomes，禁止把 extra-feedback 方法混进 same-budget 胜负。

### 8.3 代码证据

- MASA：本地 snapshot `foundations/method-wave/code/MASA/`；
- Offline-RL Harness：本地 snapshot `foundations/method-wave/code/Agentic-RL-harness/`；
- SkillAdaptor：本地 snapshot `foundations/method-wave/code/SkillAdaptor/`；
- 若作者代码不能在冻结环境运行，记录 commit、错误和最小修复；不得把自行重写结果标成 official reproduction。

## 9. 主指标与统计

### 9.1 Primary

1. target final-test success rate；
2. target normalized utility；
3. relative improvement over strongest same-budget baseline；
4. policy regret relative to per-state target action oracle；
5. Pareto frontier：success vs. tokens/steps/latency。

### 9.2 Safety/Reliability

- negative intervention rate：`G(chosen action) < G(NONE)`；
- `NONE` precision：拒绝干预的 states 中，oracle 最优是否确为 NONE；
- support coverage 与 utility 的 risk-coverage curve；
- cross-family worst-target utility。

### 9.3 Secondary Mechanism Evidence

- transported action-value MAE/sign；
- source branch effect variance；
- matched/unmatched state 分布；
- transport cost 与 action-response disagreement；
- router decision fidelity after distillation。

这些指标不能替代 primary outcome。

### 9.4 Statistical protocol

- task-paired bootstrap 95% CI；
- paired permutation/McNemar test 用于 success；
- 同时报告 absolute pp 与 relative gain；
- 预声明 pooled primary contrast 和两个 target 的分层结果；
- 多 baseline 比较做 Holm correction；
- 所有 seeds、task order、quantization 与 parser failures append-only 记录。

## 10. 必要消融

1. `branch bank -> ordinary trajectory AW`：检验同 prefix 分支是否必要；
2. `partial OT -> balanced OT`：检验 unmatched target state；
3. `response-aware metric -> semantic-only metric`；
4. `LCB policy -> posterior mean argmax`；
5. `event state -> task/category only`；
6. `structured + text -> structured-only / text-only`；
7. `multi-source -> nearest single source`；
8. `OT router -> distilled router`；
9. `one intervention -> repeated interventions` 仅作扩展，不影响主 claim。

## 11. 成功与失败标准

### Continue/claim

必须同时满足：

1. TRACE-H 在 pseudo-target pilot 上显著胜 Source-AW、Nearest-AW、MF-Gated AW、kNN 与 balanced OT 中最强者；
2. 完整实验至少一个 sealed target 上 primary utility 显著提高，另一个 target 不出现实质回退；
3. pooled primary contrast 胜最强 same-budget baseline；
4. partial transport 与 LCB 至少各有一个不可替代的 end-to-end ablation 增量；
5. target information ledger 无泄漏；
6. 主结论来自 actual task execution，不是 secondary diagnostic。

### Simplify

- kNN-Branch 与 TRACE-H 持平：删除 OT，投稿简单 branch retrieval router；
- Balanced-OT 持平：删除 partial/unmatched claim；
- Source-AW 持平：停止 policy-transport 方法主张，转为 cross-executor generalization benchmark；
- MASA/SkillAdaptor 在很少 target feedback 下远胜：将问题改为 target-budget frontier，而非宣称 zero-feedback 更优。

### Stop

- 只改善 process frequency，不改善 success/utility；
- target cross-family 明显负迁移；
- branch action effects 在 source 内不可重复；
- replay 无法恢复相同 state；
- target action outcome 在 policy seal 前泄漏；
- 最强同预算 baseline 与 TRACE-H 持平或更好。

## 12. 72 小时方法杀伤实验

### Pilot

- 环境：ALFWorld；
- source：Qwen3-4B、Qwen3-8B；
- pseudo-target：Qwen3-14B，完整留出；
- 30 source branch tasks、20 target baseline calibration tasks、30 target final-test tasks；
- actions：`NONE/CHECK/RETRY/REPLAN`；
- 每个 source episode 最多保留 2 个 branch points；
- 预计约 480 source branch continuations，加 target calibration 与各方法 final-test，合计约 750-900 runs。

### Pilot 必跑方法

`No Harness, best fixed, Source-AW, Nearest-AW, MF-Gated AW, kNN-Branch, Balanced-OT, TRACE-H`。

### Pilot 唯一 Go 条件

TRACE-H 必须在 pseudo-target end-to-end utility 上胜最强同预算 baseline，并且增量不是单一 task type 或 parser bug 造成。effect prediction 变好但 utility 不变，判定 No-Go。

## 13. 完整工作量

| 模块 | 聚焦人时 |
|---|---:|
| 环境 branch/replay 与 state hash | 25-35 |
| 统一 action semantics 与 source data | 25-40 |
| partial OT、uncertainty、router | 20-30 |
| 三个公开 baseline 适配与公平预算 | 30-45 |
| 完整运行与故障恢复 | 30-45 |
| 统计、图、论文与审计 | 25-35 |
| **总计** | **155-230** |

本方案比诊断版工作量更大，但更符合 design paper。24GB RTX 5090 Laptop GPU 只用于统一 4-bit 的 4B/8B/14B engineering micro-pilot；Qwen3-32B 与 Gemma prospective targets 不在本机试跑。导师组已确认 16 张 B200 GPU 配额，正式 source、750-900-run kill test、sealed targets 和主矩阵使用统一 BF16 runtime。若 2026-07-14 前本机 runner/branch/replay 仍未过门，不应直接用集群扩量。

本地、L4 与 B200 的具体边界、GPU-hour 包络和 Slurm 执行规则见[本地工作站与 HiPerGator 分阶段执行方案](../deployment/local-vs-hipergator-execution-plan-zh.md)。
本机逐项实验、314-run 上限与 B200 交接门见[本机具体实验计划](../experiments/local-development-experiment-plan-zh.md)。

## 14. AAAI 主张顺序

正文只按以下顺序讲：

1. **问题：** learned harness policy 在 executor 更换时失效，现有 target optimization 昂贵；
2. **方法：** branch source control outcomes，partial-transport target states，compile conservative router；
3. **胜负：** 未见 executor 上直接胜 same-budget controllers/adapters；
4. **机制：** unmatched mass 和 event-local response 为什么降低负迁移；
5. **边界：** 有 target feedback 时 MASA/SkillAdaptor/target-AW 的成本效用前沿。

不再以“我们预测得更准”作为论文高潮。论文的 killer figure 应是：

```text
Target intervention feedback budget
        x-axis: 0 -> many target action outcomes
        y-axis: end-to-end utility

TRACE-H 在 x=0 处领先同预算方法，
并以更少 target feedback 接近 target-trained upper comparators。
```

## 15. 当前结论

这次扩张不是把旧 predictor 包装成“大系统愿景”，而是实质改变研究产物、算法监督、部署形态和主评价：

```text
旧：预测每个 patch 的 aggregate effect
新：运输并执行 state-conditioned harness control policy
```

TRACE-H 现在有一个可与 Offline-RL Harness、MASA、SkillAdaptor 和 SkVM 正面比较的新机制。但它仍没有自有结果；在 72 小时 pilot 赢下 end-to-end PK 前，只能称为强设计假说，不能称为已验证方法。
