# TRACE-H 术语与符号说明：导师版

- **用途：** 不假设读者熟悉强化学习、最优传输或 agent harness。
- **方法：** 每个术语说明“是什么、在本文中具体指什么、容易与什么混淆”。
- **对应 Proposal：** [TRACE-H Policy Transport](../proposals/trace-h-policy-transport-proposal-zh.md)

## 0. 名称与研究载体

| 名称 | 含义 |
|---|---|
| **LLM** | Large Language Model，大语言模型 |
| **TRACE-H** | Trajectory-Response Alignment for Cross-Executor Harnesses；强调轨迹响应对齐与跨执行模型 harness policy |
| **AAAI-27** | 计划投稿的 AAAI 2027 主会；不是方法或 benchmark 名称 |
| **ALFWorld** | 文本交互式家务环境，agent 通过离散动作完成拾取、加热、清洁等任务；本项目主实验环境 |
| **WebShop** | 模拟网页购物的交互环境；本项目用于第二环境验证 |

## 1. 先看整体关系

```text
Agent system
  = LLM executor            负责理解当前上下文并生成下一步行为
  + harness                 模型外部的运行和控制逻辑
  + tools/environment       行为实际作用的工具与任务环境

Harness policy
  = 当前执行状态 state
    -> 选择一个 harness action
    -> NONE / CHECK / RETRY / REPLAN
```

TRACE-H 不修改 LLM 权重。它学习的是模型外部的控制策略：在什么状态下，是否应让系统检查、重试或重规划。

## 2. 系统对象

| 术语 | 准确定义 | 本项目中的例子 | 不等于什么 |
|---|---|---|---|
| **基础模型 / backbone** | 提供语言理解、推理和生成能力的 LLM checkpoint | Qwen3-8B、Gemma3-12B | 完整 agent system |
| **LLM executor** | 在给定当前消息、工具状态和 harness 指令后，实际生成下一步内容或工具 action 的基础模型实例 | 同一个 ALFWorld runner 中的 Qwen3-8B | 环境、工具或 harness 本身 |
| **Agent** | executor、harness、工具接口、环境状态与运行循环的整体 | 能观察 ALFWorld、发动作并接收反馈的完整系统 | 仅一个聊天模型 |
| **Harness** | 位于模型权重之外，负责组织观察、工具、检查、重试、记忆、终止等过程的运行层 | action parser、失败恢复、检查和重规划逻辑 | 仅 system prompt；也不限于自然语言 skill |
| **Harness action** | Harness 在一个执行状态上可采取的外部控制操作 | `CHECK/RETRY/REPLAN/NONE` | 环境动作，如 `open fridge` |
| **Harness policy** | 从当前执行状态到 harness action 的决策规则 `pi(a|s)` | 发现 invalid action 后选择 RETRY，否则 NONE | 一张静态 skill 文档或一次全局配置选择 |
| **Runtime router** | 在 agent 运行过程中反复读取 state，并调用 harness policy 选择 action 的轻量组件 | 输入错误类型和剩余预算，输出 RETRY | 只按任务标题检索 skill 的 retriever |
| **Prompt** | 一次模型调用中提供的自然语言或结构化指令 | system/user/tool messages | 完整 harness |
| **Skill** | 可检索、复用的过程知识或执行说明 artifact | “如何在 WebShop 比较商品属性”的说明 | 动态 runtime policy |
| **Static skill** | 任务开始或检索时注入，执行中内容不随 state-action response 动态改变的 skill | MASA 改写后的 skill library | TRACE-H 的 state-conditioned router |
| **Model card** | 模型架构、训练来源、能力强弱等结构化描述 | MASA 输入的 Qwen3 capability profile | 在目标任务上实际测得的 action outcomes |

### 2.1 Harness MDP

`MDP` 是 Markov Decision Process，即“状态下做动作、环境转移、最终获得回报”的顺序决策模型。本文写作：

```text
M_m = (S, A_H, P_m, R, H)
```

| 词 | 含义 |
|---|---|
| **Transition** | 在 state `s` 采取 action `a` 后，下一个 state 如何变化 |
| **Dynamics / `P_m`** | Executor `m` 与环境共同决定的 transition 规律；模型改变后，同一 harness action 的后果可能改变 |
| **Reward / utility** | 衡量任务结果及成本的数值 |
| **Return** | 从某个 state/action 开始直到任务结束得到的总 reward；本项目主要使用 terminal utility |
| **Horizon / `H`** | 一个 episode 允许的最大步骤或控制时长 |
| **Finite-horizon** | Horizon 有预声明上限，任务不会无限运行 |

把 agent 写成 MDP 是分析和设计算法的近似，不意味着文本 state 一定满足严格 Markov 性。为减少遗漏，state representation 会保留必要的局部历史和进度字段。

## 3. 来源模型、目标模型与“未见”

| 术语 | 定义 | 关键边界 |
|---|---|---|
| **Source executor** | 允许收集不同 harness actions 结果、用于学习 response 的模型 | 可以知道 CHECK、RETRY 等在来源模型上是否改善最终结果 |
| **Target executor** | 希望部署新 policy 的模型 | policy freeze 前不能查看其非 NONE action 结果 |
| **Unseen executor** | 整个模型身份在 policy learning 的 outcome supervision 中留出 | 允许公开 model metadata 和预声明 baseline trajectories；不是完全黑箱、零访问 |
| **Whole-executor holdout** | 留出的不是若干任务，而是一个完整基础模型的全部 harness-action outcomes | source 用 Qwen3-4B/8B/14B，target 可为 Qwen3-32B |
| **In-family target** | 与 source 属于同一模型家族，但尺度或 checkpoint 不同 | Qwen3-32B 相对 Qwen3-4B/8B/14B |
| **Cross-family target** | 与 source 来自不同模型家族 | Qwen3 sources 到 Gemma target |
| **Target-intervention-free** | policy freeze 前，目标模型只运行 `NONE` baseline，不运行 CHECK/RETRY/REPLAN | 不等于 target-query-free；仍允许 baseline calibration calls |
| **Target action feedback** | 在目标模型上执行非 NONE harness action 后得到的 success、reward、cost 或失败信息 | SkillAdaptor qualification rerun 会产生该反馈 |
| **Zero target action feedback** | freeze 前目标 action feedback 条数为零 | 目标 baseline observations 不是 action feedback |

## 4. Baseline、轨迹与执行状态

| 术语 | 定义 | 项目中的具体形式 |
|---|---|---|
| **Baseline harness** | 不启用待研究控制动作时的固定参考运行方式 | `A_H=NONE`，其余模型、工具、预算不变 |
| **Baseline trajectory** | 使用 baseline harness 得到的完整执行记录 | observations、LLM actions、tool results、errors、final success |
| **Target baseline calibration** | 用目标模型 baseline trajectories 估计 target state 分布，但不读取任何 intervention outcome | 20 个与 final-test 不重合的 ALFWorld tasks |
| **Trajectory** | 从任务开始到终止的有序执行序列 | `s0,a0,s1,a1,...,sH` |
| **State** | 某一时刻 router 可观察且允许使用的信息摘要 | 当前进度、最近 action/observation、错误类型、验证状态、剩余预算 |
| **Event / trigger** | 允许 router 考虑非 NONE action 的预声明时刻 | invalid action、no progress、缺少前置条件、提交前不确定 |
| **Trajectory prefix** | 从任务开始到某一 event 为止的历史 | 在第一次 invalid action 后、恢复动作前的完整记录 |
| **Snapshot** | 可直接恢复某个环境状态的保存文件 | ALFWorld environment serialization |
| **Replay** | 环境不能 snapshot 时，按原动作重新执行到相同 prefix | 只有 state hash 一致才接受 |
| **State hash** | 对关键环境状态和轨迹字段做确定性摘要，用于确认两个分支起点相同 | inventory、location、task progress 与最近 observation 的 hash |

## 5. Branch 与 Counterfactual Response

### 5.1 Branch 是什么

在同一个已保存 trajectory prefix 上，复制四个完全相同的起点，分别运行：

```text
Branch 1: NONE
Branch 2: CHECK
Branch 3: RETRY
Branch 4: REPLAN
```

每个分支继续运行到任务结束，并记录最终 success、tokens、steps 和 invalid actions。

### 5.2 为什么称为 counterfactual

现实中一个 episode 只能采取一个 action；其余 actions 是“如果当时换一种控制会怎样”。本项目不是让 LLM 口头猜测这些反事实，而是从同一个保存状态实际执行每个分支，因此获得实验性 counterfactual outcomes。

这里的 counterfactual 仍有边界：不同分支后的 LLM 生成可能有随机性，所以需要固定采样协议、重复运行和不确定性估计；不能把一次分支差异宣称为无条件因果定律。

| 术语 | 定义 |
|---|---|
| **Terminal return / `G_i(a)`** | 从第 `i` 个 prefix 采取 action `a` 后，直至任务结束得到的总效用 |
| **Action response** | 某 action 在一个 state 上造成的最终结果变化 |
| **Action advantage / `A_i(a)`** | `G_i(a)-G_i(NONE)`；大于零表示该 action 比不干预更好，小于零表示有害 |
| **Event-level** | 估计单位是具体执行事件，不是整个模型或任务类别的平均值 |
| **Counterfactual Branch Bank** | 所有 source prefixes、state features、分支 outcomes、advantages、成本和方差组成的数据集 |
| **Response bank** | Counterfactual Branch Bank 的简称；强调其中保存了 action response |

## 6. 四个 Harness Actions

| Action | 含义 | 允许做什么 | 不允许做什么 |
|---|---|---|---|
| **NONE** | 不增加控制 | 让 executor 按 baseline 继续 | 偷偷加入提示或额外调用 |
| **CHECK** | 检查当前动作所需的状态或前置条件 | 查看 inventory、确认工具返回、核对提交条件 | 直接提供任务答案或完整专家轨迹 |
| **RETRY** | 对刚刚失败或无效的 action 做一次规范化重试 | 修正格式、参数或可恢复的调用错误 | 对正常 action 无限重试 |
| **REPLAN** | 根据当前状态生成短的剩余步骤计划 | 最多三步、随后返回 executor | 用完整外部 planner 接管整个任务 |

这些 actions 必须跨 executor 保持相同语义和预算，否则测到的是不同实现，而不是 policy transport。

## 7. “Transport”到底运输什么

### 7.1 Policy transport

运输的是来源 states 上学到的 `action -> outcome` 经验，并据此构造目标 `state -> action` policy。它不复制模型权重，也不简单复制一段 prompt。

### 7.2 State representation

把一个复杂 prefix 转为可比较向量：

```text
z = [event type,
     progress and remaining budget,
     verifier/tool flags,
     recent action outcome,
     local text embedding]
```

- **结构特征：** 可程序读取的离散/数值字段，如错误类型、步数、是否有工具失败。
- **语义特征：** 最近 observation/error 和当前 subgoal 的文本 embedding。
- **Embedding：** 把文本映射到数值向量；只用于比较局部语义，不代表模型真实内部思维。

### 7.3 Optimal Transport, OT

最优传输原本研究如何以最小代价把一个分布的“质量”匹配到另一个分布。这里把 source branch states 看作已有 response 标签的点，把 target baseline states 看作没有 response 标签的点；OT 寻找总体代价最小的软匹配。

| 术语 | 本项目中的含义 |
|---|---|
| **Transport cost `C_ij`** | source state `i` 与 target state `j` 的不相似度；由结构距离、语义距离和 event compatibility 构成 |
| **Transport plan `Gamma`** | 每个 target state 从哪些 source states 借多少证据的权重矩阵；不是物理搬运 |
| **Mass** | 一个 state 在分布中需要被匹配的统计权重；不是内存大小 |
| **Balanced OT** | 强制 source 和 target 的全部 mass 都完成匹配，即使某些状态并不相似 |
| **Unbalanced OT** | 允许边际质量发生变化，对不可靠匹配付惩罚而非强制匹配 |
| **Partial OT** | 只运输有足够支持的一部分 mass，其余部分可不匹配 |
| **Unmatched mass** | target state 没有可靠 source 对应时保留的未匹配权重；router 对其倾向选择 NONE |
| **Target-private state** | 目标模型出现、source response bank 中没有可信对应的执行状态；这里的 private 是“目标独有”，不是隐私数据 |
| **Response-aware** | state metric 和超参数通过 source leave-one-executor-out 的 action-response utility 选择，而非只看文本相似度 |

## 8. 从运输结果到可执行决策

| 术语 | 定义 | 直观解释 |
|---|---|---|
| **Transported action advantage** | 按 `Gamma` 加权 source advantages，得到目标 state 上各 action 的估计收益 | 相似来源事件表明 CHECK 平均有多大帮助 |
| **Uncertainty** | 来源分支随机性、不同 source executors 分歧和 transport 匹配不确定性的合计 | 估计值为正不代表足够可靠 |
| **Confidence interval** | 根据重复/重采样得到的 action advantage 可能范围 | 例如 CHECK 收益区间 `[-0.02, 0.10]` |
| **LCB** | Lower Confidence Bound，下置信界；区间的保守下端 | 只有 LCB 仍大于成本时才干预 |
| **Support coverage** | target state 获得了多少可靠 source mass | coverage 太低意味着来源经验不足 |
| **Conservative policy** | 证据不足时选择 NONE，而不是强制干预 | 优先避免负迁移 |
| **Policy compilation** | 把冻结后的 features、transport weights、thresholds 和 action rules 序列化为可加载 router artifact | 这里不是编译机器代码，也不同于 SkVM 的 skill compilation |
| **Router artifact** | 可版本化、可 hash、可直接加载的 policy 文件 | 包含 schema、prototypes、weights、thresholds 和 fallback |
| **Policy distillation** | 用小模型模仿原 OT router 的决策，以降低运行时计算 | 不使用 target action rewards 训练 |

## 9. 评价指标

| 术语 | 定义 | 注意点 |
|---|---|---|
| **Success rate** | final-test tasks 中成功任务比例 | 最直观的主指标 |
| **End-to-end** | 从任务输入到最终环境结果完整运行，而非只评价推荐或预测 | 包含 action 对后续轨迹的真实影响 |
| **Normalized utility** | 成功收益减去预声明 token、step、invalid-action 成本 | 必须同时报告 success，不能靠权重掩盖失败 |
| **Negative intervention** | 所选非 NONE action 的真实 return 低于同 state 的 NONE | 表示“帮倒忙” |
| **Oracle** | 事后知道 target 每个 action 真值、总能选最优 action 的评估上界 | 只用于分析，部署时不可访问 |
| **Policy regret** | Oracle action return 减去实际 policy action return | regret 越小越好；不是情绪意义的“遗憾” |
| **Same-budget baseline** | source data、target baseline、target action feedback 和测试额外成本均与 TRACE-H 对齐的方法 | 只对齐模型调用次数仍不够 |
| **Target-feedback frontier** | 横轴为目标 action outcomes 数量，纵轴为 end-to-end utility 的曲线 | 公平比较零反馈和 active adaptation |
| **Primary metric** | 决定论文主要成败、事前冻结的指标 | success、utility、policy regret |
| **Secondary metric** | 用于解释机制但不能单独支撑论文的方法指标 | effect MAE、OT cost、coverage、calibration |
| **Risk-coverage curve** | policy 越保守时，覆盖多少 states 与负干预风险之间的曲线 | 展示 NONE/abstention 的代价和收益 |
| **Calibration** | 预测置信度与真实正确频率是否一致 | 例如声称 80% 可靠的 actions 是否约有 80% 真正有益 |
| **MAE** | Mean Absolute Error，预测 action advantage 与真实 advantage 的平均绝对差 | 只能作为 secondary mechanism metric |
| **Pareto frontier** | 在 success 与成本等多个目标间，不存在“所有方面都更好”的方法集合 | 用于比较高成功高成本与低成本稍低成功的方案 |
| **Ablation** | 从完整方法中删除或替换一个组件，观察结果下降多少 | partial OT 改为 balanced OT，检验 unmatched mass 是否必要 |

## 10. 方法比较术语

| 名称 | 简明定义 | 在 PK 中回答什么 |
|---|---|---|
| **Offline RL** | 只使用预先收集的轨迹训练 policy，不在训练时继续探索 | 固定 executor 上能否学 controller |
| **RL** | Reinforcement Learning，根据 state、action 和 return 学习决策 policy | 本文不训练基础模型权重，只学习外部 harness control |
| **AW / Advantage-Weighted Regression** | 更高 return 的 state-action 样本获得更大模仿权重 | Source-AW 是直接 learned-controller baseline |
| **Source-AW** | 在 pooled source executors 上训练 AW policy，原样用于 target | 不做运输是否已经够好 |
| **Behavioral fingerprint** | 从 baseline success、cost、错误和输出分布形成的模型行为摘要 | 不运行目标 harness actions 时比较 source 与 target |
| **Nearest-AW** | 根据 target baseline behavioral fingerprint 选择最相似 source policy | 简单 source selection 是否够好 |
| **Metric Freedom / MF** | 从 repeated baseline runs 的输出分布估计通用 skill headroom | 只能判断整体是否有受益空间 |
| **Generic headroom** | 某 task/metric 上“增加外部帮助可能带来多少总体收益”的空间 | 对同一 state 的不同 actions 不提供细粒度排序 |
| **MF-Gated AW** | MF 只决定启用或禁用 Source-AW | generic headroom 能否替代 state-level transport |
| **kNN-Branch** | 直接使用最近 `k` 个 source branch states 的 response | OT 是否比最近邻真正有增量 |
| **Balanced-OT** | 所有 target states 被强制匹配的 OT baseline | unmatched mechanism 是否必要 |
| **PAR-style penalty** | 根据 source-target representation mismatch 降低不匹配 source transitions 的权重 | partial OT 是否优于一般 dynamics mismatch 修正 |
| **MASA** | 使用 target model card、目标 rollout feedback 搜索/改写静态 skills | model-aware static adaptation 强方法 |
| **DS-Adapter** | MASA 中由强 teacher 根据 target model card 做一次 skill rewrite、不做迭代搜索的 baseline | 与零 target action reward 的 TRACE-H 比较 model-aware static rewrite |
| **SkillAdaptor** | 根据目标失败轨迹修改 skill，并通过目标重跑决定是否接受 | active target feedback 强方法 |
| **SkVM-style AOT/JIT** | AOT 按 target capability 编译 skill；JIT 根据运行失败继续重编译 | capability compilation 与 policy transport 的差别 |
| **AOT** | Ahead-of-Time；在正式执行前根据 target capability 编译 artifact | SkVM 的 install-time 阶段 |
| **JIT** | Just-in-Time；执行过程中根据重复失败或稳定模式继续优化 | 会读取运行时 target feedback |
| **UCB** | Upper Confidence Bound；在搜索中兼顾当前高分候选与尚未充分尝试候选 | MASA task-specific tree search 使用 UCB1 |
| **QoS** | Quality of Service；成本、延迟、风险、可靠性等部署约束 | SkillSelect-Serve 的 bundle 选择维度 |

## 11. 实验与统计术语

| 术语 | 定义 |
|---|---|
| **LOMO / leave-one-model-out** | 每次把一个 source model 当伪目标，其余 source 训练，用于冻结超参数 |
| **LOEO / leave-one-executor-out** | 更准确的本项目说法；留出整个 executor 的 action outcomes |
| **Seal / freeze** | 在查看 target action outcomes 前，把代码、数据 hash、policy、baseline 和统计规则固定下来 |
| **Unseal** | freeze 后第一次运行或读取 target 非 NONE action outcomes |
| **Leakage** | target action outcome 在 freeze 前影响了任何方法或选择 |
| **Target Information Ledger** | 逐方法记录见过多少 target baseline/action outcomes 和测试成本的账本 |
| **Task-paired comparison** | 每个方法在同一 task 上比较，减少任务难度差异 |
| **Bootstrap interval** | 对 tasks 重采样，多次重算差值，得到不确定区间 |
| **McNemar test** | 比较两个方法在同一批二元成功/失败任务上的差异 |
| **Holm correction** | 同时比较多个 baselines 时控制多重检验假阳性 |
| **CI** | Confidence Interval；本文主要使用 task bootstrap 形成差异区间 |
| **KL divergence** | 衡量两个概率分布差异的非对称量；在 unbalanced OT 中作为边际偏离惩罚 |
| **Entropy regularization** | 鼓励 transport plan 不过度尖锐、同时使优化可高效求解的平滑项 |

## 11.1 工程和实验文件术语

| 术语 | 定义 |
|---|---|
| **Checkpoint** | 一套冻结模型权重及配置的具体版本；同一模型名的不同 checkpoint 不能混用 |
| **Quantization** | 用较低数值精度存储/运行模型以降低显存，如 4-bit；可能改变行为，必须冻结 |
| **Chat template** | 把 system/user/tool messages 格式化为模型实际输入 token 序列的规则 |
| **Seed** | 控制可控随机过程的初始值；并不保证所有远程 LLM 完全确定 |
| **Runner** | 执行 task、调用 executor/tools、记录 trajectory 和结果的程序 |
| **Manifest** | 机器可读的任务、模型、split、顺序和配置清单 |
| **Artifact** | 方法生成并保存的可复用文件，如 router、skill JSON 或 policy weights |
| **Parser** | 把模型文本转成环境可执行 action 的程序；解析失败计入真实失败 |
| **Smoke test** | 在少量 tasks 上确认环境、模型和日志链路能跑通的快速测试 |
| **Pseudo-target** | Pilot 中临时按正式 target 规则完全留出的 source model，用于提前检验方法 |
| **Pseudo-label** | 在没有真实 target action reward 时，由冻结方法生成的训练/蒸馏标签 |
| **Pooled source** | 把多个 source executors 的数据合并训练一个方法 |
| **PK / method comparison** | 与其他方法在共同任务和预算下正面比较；不是只在 related work 中讨论 |

## 12. 数学符号

| 符号 | 含义 |
|---|---|
| `m` | 一个 LLM executor |
| `S` | 可观察 state 空间 |
| `A_H` | Harness action 集合 `{NONE,CHECK,RETRY,REPLAN}` |
| `P_m` | executor `m` 与环境共同产生的 transition dynamics |
| `R` | terminal task reward/utility |
| `H` | 最大时域或最大步骤数 |
| `pi_T(a|s)` | 目标 executor 在 state `s` 选择 action `a` 的 policy |
| `G_i(a)` | 第 `i` 个 prefix 采取 action `a` 后的 terminal return |
| `A_i(a)` | `G_i(a)-G_i(NONE)`，action advantage |
| `z_i` | state/prefix 的数值 representation |
| `C_ij` | source state `i` 与 target state `j` 的 transport cost |
| `Gamma_ij` | target state `j` 从 source state `i` 接收的证据权重 |
| `LCB_T(j,a)` | target state `j` 上 action `a` 的保守收益下界 |
| `J_T(pi)` | policy `pi` 在 target executor 上的期望 end-to-end utility |
| `lambda_*` | success 与 token/step/invalid-action 成本之间的预声明权重 |

## 13. 一个完整例子

任务：在 ALFWorld 中把苹果加热后放到桌上。

1. Qwen3-8B 尝试 `heat apple with microwave`，但苹果还不在 inventory，环境拒绝该 action。
2. 这一时刻形成 event state：`invalid action + missing precondition + medium budget`。
3. Source branch bank 从同一 prefix 分别执行 NONE、CHECK、RETRY、REPLAN。
4. CHECK 先核对 inventory 并引导拾取苹果，最终成功；NONE 和 RETRY 继续失败；REPLAN 成功但多耗步骤。
5. 因而该 source state 上 CHECK 的 advantage 为正，RETRY 为负。
6. 未见目标模型 Gemma 在 baseline calibration 中也出现“动作被拒绝、物品不在 inventory”的 state，但从未运行过 CHECK。
7. Partial OT 将它的大部分 mass 对齐到来源的 missing-precondition states，同时保留不确定部分 unmatched。
8. TRACE-H 估计 CHECK 的 LCB 扣除成本后仍为正，于是 runtime router 选择 CHECK。
9. Final test 中实际运行 CHECK；如果任务成功且 utility 高于 Source-AW、kNN、Balanced-OT 等方法，这才是支持 TRACE-H 的 primary evidence。

这个例子概括了整个方法：来源侧实际分支获得 response，目标侧只观察 baseline state，运输后执行 policy，最后用真实任务结果而不是诊断分数判断胜负。

## 14. 后续写作必须统一的叫法

| 应使用 | 不要混用 | 原因 |
|---|---|---|
| **Executor / 执行模型** | agent、system、backbone 随意互换 | Executor 特指 agent loop 中产生下一步行为的模型实例 |
| **Agent system** | LLM | Agent 还包括 harness、tools 和 environment |
| **Harness action / 控制动作** | environment action | CHECK 与 `open fridge` 属于两个动作空间 |
| **Policy transport / 策略运输** | prompt transfer、skill copy | 本文运输的是 state-action response |
| **Target baseline observation** | target feedback | Baseline state 可见，但非 NONE action outcome 不可见 |
| **Action response/advantage** | relevance、utility label | Response 来自实际 branch terminal outcome |
| **NONE** | abstain、reject 随意互换 | NONE 是一个可执行动作，表示保持 baseline；不是拒绝输出预测 |
| **Harness action** | patch | 当前方法研究运行时 actions；patch 是旧 Proposal 的静态干预单元 |
| **Policy compilation** | SkVM compilation | 本文是冻结并序列化 router；SkVM 是 capability-based skill/code compilation |
| **Target-private state** | private data | 仅表示目标分布独有、来源无对应 |
| **Primary execution result** | diagnostic result | 论文胜负只由真实 success/utility/regret 决定 |

正式 Proposal、代码 schema 和实验表应沿用这一词表。若方法对象改变，先更新词表与 decision record，再修改正文，避免同一个词在不同文档中指向不同对象。
