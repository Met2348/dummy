# DR-0002：TRACE-H Idea 重构与 AAAI-27 再评估

- **日期：** 2026-07-11
- **负责人：** 项目组
- **状态：** 已被 DR-0003 的第二轮碰撞复核取代；保留为第一轮决策快照
- **复审日期：** 72 小时 pilot 完成后，最晚 2026-07-14
- **替代：** DR-0001 中的 FORECAST-H 方法中心与评分
- **保留：** DR-0001 的 deadline 事实、证据缺口和密封实验纪律仍有效
- **全文复核：** [SEAGym、LIFE-HARNESS 与九篇近邻逐篇审计](../foundations/notes/fulltext-collision-reassessment-zh.md)

> **后续更新：** 第二轮全文检索发现 Metric Freedom 已占据宽泛的 baseline-only a priori skill-utility prediction。当前 claim、评分、接受概率与实验门以 [DR-0003](0003-metric-freedom-collision-reassessment-zh.md) 为准；本记录不做事后重写。

## 结论摘要

**TRACE-H 是比旧版 FORECAST-H 更强、更整洁的 idea，但项目仍然没有一篇论文。**

重构把高维 target-effect regression 换成了由 event-gated patch 契约导出的 trigger-conditioned standardization：目标 baseline 轨迹提供失效机会分布，来源 paired experiments 提供条件 rescue/harm profile，二者组合预测密封目标效应。它显著改善 neatness、soundness、可解释性和三周可执行性，也更容易在七页内讲清。

新增检索后又逐页复核了 SEAGym、LIFE-HARNESS 与九篇最接近工作。修订后的判断不是“碰撞太多所以绕开”：SEAGym 用跨 backend 矩阵和 update artifact case study 提出了有依据但仍属事后的 failure-surface alignment explanation；LIFE-HARNESS 则证明一套环境专用 bundled harness 可以从一个来源模型广泛迁移到 17 个其他模型。两者共同建立了 TRACE-H 必须正面检验的经验张力：**迁移通常可以成功，但何时无益、何时反号，尚不能在目标 intervention 运行前判断。** 新颖性不包括 H-FS 定性假说本身，而在于对 dormant atomic patch 做 baseline-observed、intervention-outcome-sealed 的跨模型 signed-effect prediction，并把 opportunity shift 与 conditional response shift 分开。

当前决定仍是：**只投入 72 小时验证核心机制；pilot 未通过时立即转向或停止，不为 AAAI deadline 强行制造故事。**

## 1. 决策对象

本记录回答：

> 将研究中心从通用 FORECAST-H 改为 TRACE-H 后，idea 的 neatness、excitement、证据强度、工作量、新颖性、社区接受概率与 AAAI 审稿维度是否实质改善，是否值得继续冲 AAAI-27？

正式方案见 [TRACE-H 正式 Proposal](../proposals/trace-h-formal-proposal-zh.md)，候选比较见 [Idea 大调整记录](../notes/idea-iteration-and-selection-zh.md)，全文证据边界见[碰撞复核](../foundations/notes/fulltext-collision-reassessment-zh.md)。

## 2. 截止日期与硬约束

AAAI-27 官方 timetable：

- 2026-07-21：摘要截止；
- 2026-07-28：正文截止；
- 2026-07-31：supplementary 与 code 截止；
- 所有时间为 UTC-12。

从 2026-07-11 起，只有约 10 天到摘要、17 天到正文。[AAAI-27 Main Technical Track](https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/) 强调 significance、novelty、theoretical/empirical soundness、relevance、clarity、responsible research 和 reproducibility，并将主内容限制为七页。这个版面要求 idea 能被一张图和一个核心式解释。

当前工程硬事实：

- 相关文献与 protocol 资产充分；
- Harness-Bench 106 个任务的代码已本地快照；
- ToolBench-X 所称 code/data 当前尚未真正发布，不能依赖；
- 本项目尚无 reference harness、patch 实现、pilot trajectory、source matrix 或 target seal；
- 四个稳定 model endpoints 和可用额度尚未在本项目中验证。

## 3. 变化前后对比

| 项目 | FORECAST-H | TRACE-H | 影响 |
|---|---|---|---|
| 方法中心 | 高维 target effect predictor | trigger-conditioned standardization | 更低自由度、更可解释 |
| Target features | 模型接口、任务、预算、行为 fingerprint | patch-specific baseline failure surface | 机制联系更直接 |
| Intervention 范围 | 3-5 个不同机制、含 prose | 3 个 dormant event-gated patches | 原子性更强 |
| 关键假设 | 某组特征可预测 delta | 条件 rescue/harm 比 raw delta 更稳定 | 更清楚、更可证伪 |
| 决策动作 | reuse/retune/reject | deploy/reject/abstain | 去掉未实现的 retune |
| 风险表述 | risk-controlled 倾向 | empirical risk-coverage | 避免无依据理论保证 |
| 实验解释 | 通用 transport law | 指定 patch family 的 prospective audit | 主张更克制 |
| 核心图 | predicted vs observed | failure surface x rescue profile + sealed result | narrative 更完整 |

## 4. 总评分

评分区分“当前 idea 质量”“成功实验后的论文上限”和“当前投稿准备度”。

| 维度 | 当前 idea | 成功实验上限 | 置信度 | 判断 |
|---|---:|---:|---|---|
| Neatness | 8.6/10 | 9.2/10 | 中高 | 一个契约、一个分解式、一个 sealed test |
| Excitement | 8.2/10 | 9.0/10 | 中 | 能把 model upgrade 风险变成可预判决策 |
| 问题证据 | 9.3/10 | 9.3/10 | 高 | 正负迁移、交互和 verifier tax 证据充足 |
| 机制合理性证据 | 7.8/10 | 8.7/10 | 中 | SEAGym 的 failure-surface case study、LIFE-HARNESS 的广泛迁移与反号近邻形成互补证据，但非直接 sealed 验证 |
| 本项目方案证据 | 1.0/10 | 8.5/10 | 高 | 当前仍为零结果 |
| Novelty | 7.8/10 | 8.6/10 | 中 | SEAGym 提出 H-FS 但没有定量 surface、atomic effect、target seal 或 predictor；完整 TRACE-H 仍须证明条件响应增量 |
| Technical depth | 7.8/10 | 8.6/10 | 中 | contract、estimand、standardization、uncertainty、decision |
| Soundness potential | 8.1/10 | 9.0/10 | 中高 | 比高维小样本 regression 更可信 |
| 工作量可行性 | 5.5/10 | 7.0/10 | 中 | 576 episodes，仍受 runner/endpoint 支配 |
| AAAI community fit | 8.1/10 | 8.8/10 | 中 | agent reliability + OOD effect prediction + selective deployment |
| Clarity potential | 8.7/10 | 9.2/10 | 高 | 七页内可形成单中心叙事 |
| Reproducibility potential | 8.8/10 | 9.4/10 | 高 | 离线任务、contracts、hash、append-only traces |
| 当前投稿准备度 | 2.0/10 | 8.3/10 | 高 | 文献和 proposal 不是实验结果 |

**当前 idea 质量：约 8.2/10，高于旧版约 7.4/10。**

**当前 AAAI-ready paper 质量：仍约 2/10。** idea 变好不等于结果已经存在。

## 5. Neatness 详评

### 为什么更 neat

核心逻辑可以压缩为：

```text
target patch effect
  = target baseline trigger distribution
  x source conditional rescue/harm profile
```

每个名词都对应可记录的实验量：trigger、rescue、harm、cost、sealed effect。相比“模型特征可能预测效果”，它不需要审稿人相信隐藏 embedding 或复杂交互项。

Dormant contract 还提供一个非常干净的 falsification test：no-trigger task 理论上不应改变。若改变，就是实现不原子，不是模型学术结果。

### 仍不够 neat 的部分

- 三个 patch 的 trigger taxonomy 可能膨胀；
- baseline success、trigger depth、remaining budget 都加入会再次变成特征工程；
- 目标 task-level baseline 与 target aggregate effect 的关系必须讲清；
- success 与 cost 若同时做主结果，会分散中心。

### 强制约束

- 主模型最多使用 trigger type、baseline status 和一个 depth bin；
- success delta 为主，cost-adjusted utility 为次；
- 只保留一个 reference harness 与一个 task suite；
- retune、自动 patch 生成和第二 benchmark 不进正文主线。

## 6. Excitement 详评

### 兴奋点

1. **真实部署问题：** 模型版本更新频繁，harness patch 通常被默认沿用，但重新全量 sweep 昂贵。
2. **反直觉结果空间：** 更强模型可能更少触发 recovery，却在触发时更能利用或更被干预伤害，总体 delta 不随 capability 单调变化。
3. **可展示的 prospective moment：** 先 commit 目标效应和部署决策，再打开 target patch outcomes，可信度高于事后解释。
4. **无论结果方向都有科学意义：** 若 conditional response 稳定，可形成 transport principle；若不稳定，则说明 baseline-only failure type 不足以迁移，必须主动 probe。

### 兴奋度下降条件

- 所有 patch 在所有模型上都只正不负；
- source mean 已准确预测全部 target effects；
- 触发只发生在极少任务；
- 结果仅是“错误多的模型更需要 retry”；
- 所谓机制只重述 baseline failure rate，而没有条件 response 分析。

### 达到 9 分所需结果

至少满足其一：

- 一个 raw source-copy 或 failure-surface-nearest 会做错的 sign reversal 被 TRACE-H 提前预测；
- 相对最强 simple baseline，sealed-target decision regret 明显下降；
- 发现条件 rescue profile 跨模型稳定而总体 effect 大幅变化的简洁规律；
- 能准确区分高 opportunity 但低 recoverability 与低 opportunity 但高 recoverability。

## 7. 证据强度详评

### A 级：问题存在

现有语料与新增文献共同证明：

- complete harness configuration effect 大；
- LIFE-HARNESS 的环境专用 bundled harness 可从 Qwen3-4B 广泛迁移到 17 个其他 backbone，但不是所有单元都改善；
- atomic component 可有干扰和符号变化；
- skill/artifact transfer 可正也可负；
- verifier 能频繁触发但 recovery 与安全成功有限；
- targeted diagnosis 往往比无信息 extra compute 更有效。

问题不是从空白处编造的。

### B 级：机制有合理依据

SEAGym 的全文 case study 显示，不同 rollout backend 暴露不同 failure surface，并诱导 AHE 编辑不同的 harness subsystem；其跨 backend/域 gain 只有在评测轨迹仍出现相似模式时才更稳定。LIFE-HARNESS 则说明环境侧结构可以形成很高的正迁移先验。HASP 显示 trigger 高度集中且随难度变化；The Verifier Tax 分别测量 intervention frequency、post-block recovery 与 cost；ToolBench-X 显示 hazard type 和 diagnosis 影响 recovery。它们共同支持将 opportunity 和 conditional response 分开。

但 SEAGym 的解释是事后、定性和 bundled 的，LIFE-HARNESS 也在所有目标上实际运行了整套 harness；没有一篇直接证明“source conditional rescue profile 可标准化到 sealed new model”。因此机制证据提高到 7.8，但仍只能给 B，而不是 A。

### F 级：项目自己的解法

当前尚无：

- no-trigger invariance test；
- 两模型 trigger/rescue 表；
- LOMO predictor；
- sealed target；
- regret 曲线；
- 真实负迁移被提前拒绝。

所以项目自身方案证据仍为 F/未测试。正式对老师汇报时必须把“文献支持问题”与“我们证明方案”分开。

## 8. Novelty 详评

### 不再新颖的部分

- harness effect measurement；
- model-harness interaction；
- failure-surface similarity 解释 harness transfer；
- event-gated executable intervention；
- contract-based skill；
- cross-model artifact reuse；
- fault-conditioned recovery analysis；
- held-out model absolute score prediction；
- abstention 和 risk-coverage。

### 可守住的精确新颖性

> 对满足 dormant contract 的 atomic harness patch，在完全隐藏 target patch outcomes 的条件下，用 target baseline failure surface 与 source conditional rescue/harm profile 预测 task-level 和 aggregate signed effects，并评价 deploy/abstain regret。

这个贡献应被描述为对三层假说的前瞻区分，而不是“再次发现 failure surface”：

1. `H-FS`：surface alignment 可能影响迁移，来自 SEAGym；
2. `H-OPP`：target opportunity reweighting 是否足以预测 atomic effect；
3. `H-RESP`：typed conditional rescue/harm 是否在 H-FS/H-OPP 之上提供必要增量。

### 全文复核后为什么是 7.8，而不是 7.3 或 9.0

- 核心公式是全期望与标准化，不是新数学定理；
- SEAGym 已用跨模型实验、符号反转和 update artifact case study 支持 H-FS，因此定性思想不能归本项目；
- 但 SEAGym 没有预注册 taxonomy、定量 distance、atomic treatment、target outcome seal、effect predictor 或 deployment regret，不能被写成 exact method collision；
- LIFE-HARNESS 已证明广泛跨模型 transfer，但对象是环境专用 bundled harness，所有目标 outcome 均已观察；
- 多篇 2026 工作已经覆盖其中两到四个 ingredient；
- ToolBench-X 已有五模型 targeted hint 结果，虽未预测 transfer，但很接近；
- 同期论文可能在投稿前出现；
- 审稿人可能把 conjunction 视为“合理但自然的下一步”。

### 如何回升到 8.4-8.6

不是增加模型复杂度，而是提供：

1. 真正 prospective 的目标预测；
2. 一个 source-copy、failure-surface-nearest 与 opportunity-only 都错误、TRACE-H 正确的决定性 case；
3. 条件 rescue 稳定性的系统证据；
4. 完整 contracts、traces 和 leakage audit；
5. 比 capability-only、generic ridge 与 failure-surface-nearest 都强的结果。

## 9. Technical Depth 与 Soundness

### 足够技术性的原因

- 明确的 task-level paired estimand；
- dormant contract 和 no-trigger invariance；
- rescue/harm/unchanged 多项条件模型；
- source-to-target standardization；
- cluster bootstrap 与 leave-one-model-out；
- prospective seal；
- deploy/abstain decision regret。

这是一条完整的方法链，而不是“跑 benchmark + 看热图”。

### 最大 soundness 风险

1. **模型样本只有四个：** 不能把结果解释为模型总体规律。
2. **同任务 baseline 可见：** 设置是 transductive，不是 zero-shot target transfer。
3. **条件 strata 稀疏：** 过细会严重方差膨胀。
4. **Trigger 后轨迹分叉：** 第一次 trigger 分解成立，但后续多次 activation 不能简单当独立事件。
5. **Patch 可能改变预算：** 必须同时报告 success 和 resource cost。
6. **Provider 非确定性：** temperature 0 不等于绝对可复现；没有共享 seed 时，任务级 no-trigger equality 不可直接观察。
7. **Task selection bias：** 不能用 pilot activation 挑主任务。

### 对策

- 将主张限定为四模型、三 patch、一个 task suite 的 prospective audit；
- 只对第一次 trigger 建主分解；
- 预声明 coarse strata 与回退层级；
- 任务在模型结果前冻结；
- 保存 request ID、版本、raw messages、patch diff 和 oracle artifact；
- 优先选择可固定 seed/replay 的 endpoint；否则预声明重复并把结论降为分布层效应；
- 不使用 formal causal 或 conformal guarantee 用语。

## 10. 工作量评估

### 最小主会设计

- 3 source + 1 sealed target；
- Harness-Bench 固定 36 任务；
- baseline + 3 patches；
- 576 个 primary episodes；
- 最多约 20% 边界/非确定性重复；
- 3 次 source LOMO + 1 次 prospective audit。

### 人时

正式 Proposal 估计 80-130 个聚焦人时，低于旧版 94-166 小时，但仍接近两周全职高强度工作。AI assistance 可显著承担：

- adapter、schema、unit tests、分析脚本；
- 日志归一化与表格生成；
- bootstrap、图表和文稿初版；
- 文献更新与 claim checklist。

AI assistance 不能消除：

- API/CLI 故障和额度；
- task sandbox 平台问题；
- 原子性与 leakage 判断；
- 结果异常的人工 trajectory audit；
- 作者对数字、引用和政策的责任。

### 可行性条件

- **已有稳定 reference harness + 四 endpoint：** 困难但可冲，完成强版本约 35-55%。
- **只有 endpoint，runner 从零：** 高风险，完成强版本约 15-30%。
- **endpoint 或 benchmark 到 7/14 仍不通：** AAAI 强版本低于 15%，应转后续 venue。

以上是主观条件概率范围，不是统计频率。

## 11. 社区接受概率

### 为什么可能被接受

- 直接解决 agent evaluation 中 model 与 harness 纠缠的问题；
- 从描述 heterogeneity 前进到 prospective decision；
- method 简洁、可审计、容易复现；
- 同时连接 agent systems、OOD generalization、effect estimation 和 selective prediction；
- sealed target 在快速变化领域具有额外可信度；
- 结果对实践者有明确 deploy/abstain 含义。

### 为什么可能被拒绝

1. **“只是全期望公式。”** 没有足够经验增益或新理论。
2. **“四个模型不足。”** conditional stability 可能只是挑中的模型偶然成立。
3. **“只在一个 benchmark。”** failure surface 可能是 Harness-Bench 特性。
4. **“Patch 人为。”** 三个 intervention 不能代表 harness ecosystem。
5. **“Target 不够 unseen。”** 同任务 baseline 已被观察。
6. **“预测器并不需要。”** source mean、baseline success 或 failure-surface-nearest 已同样好。
7. **“近邻太多。”** SEAGym 已提出 H-FS，LIFE-HARNESS 已给出强 cross-model transfer，HASP、ContractSkill、ToolBench-X、Harness-Bench 又覆盖大量 ingredient；若协议和结果没有独立增量，审稿人会视为自然下一步。
8. **“主张与数据不匹配。”** 从 36 任务外推到生产 model upgrade。
9. **“成本收益不清。”** 补丁提高 success 但 token/latency 更差。
10. **“复现实验依赖 API。”** 模型版本与供应商行为会漂移。

### 条件接受概率

以下是基于 paper outcome 的主观区间，不是精确预测：

| 投稿时状态 | AAAI-27 接受概率估计 | 原因 |
|---|---:|---|
| 当前 proposal，无项目结果 | 0-3% | 不是完成论文 |
| 只有 pilot，描述 trigger/heterogeneity | 4-10% | 与已有工作高度重叠 |
| 完整矩阵，但所有 target-aware 方法与 source mean 持平 | 3-8% | failure surface 没有预测或决策增量 |
| failure-surface-nearest / opportunity-only 胜 source-copy，sealed target 也成立，但 TRACE-H 无额外增量 | 12-22% | 可收缩为 H-FS/H-OPP 的简洁前瞻验证；方法新颖性较低但协议仍有价值 |
| LOMO 有提升，sealed target 弱或方向混合 | 10-20% | 有方法但主要证据不强 |
| sealed target 明显胜 simple baselines，含 failure-surface-nearest 与 opportunity-only，regret 降低，机制清楚 | 28-42% | 高于一般 base rate，但样本、单任务集与 ingredient novelty 仍有风险 |
| 上述结果再加第二模型/第二任务集的独立复现 | 35-50% | 外部有效性显著增强，但本 deadline 基本不现实 |

重构提高的是“强结果出现时的 paper ceiling”，没有自动提高“按期得到强结果”的概率。当前 unconditional chance 仍受工程完成率与未知实验结果双重折损，不能直接等同于 28-42%。

用于资源决策的当前无条件接受概率主观估计仍为 **5-12%**：全文重读改善了 idea 边界，却没有生成任何项目数据。该区间综合了 17 天内完成 runner、获得足够 trigger、方法胜过强基线、按时形成完整论文以及通过评审等连续风险；它不是统计推断，也不是 AAAI 官方 base rate。72 小时 gate 后必须按真实工程进度和 pilot effect 重估。对外汇报时应同时给出“当前无条件 5-12%”与“若 sealed target 强胜全部简单层级则条件接受 28-42%”，不能只报后者。

## 12. 模拟审稿分布

采用粗略 1-10 分，假设写作合格：

| 维度 | 弱完成稿 | 目标完成稿 | 关键分界 |
|---|---:|---:|---|
| Significance | 6 | 8 | 是否解决实际 model upgrade decision |
| Novelty | 5 | 7-8 | 是否明显超越直接邻居 |
| Soundness | 5 | 8 | contract、seal、paired statistics 是否完整 |
| Technical quality | 5 | 8 | 条件标准化是否真正优于简单基线 |
| Empirical evaluation | 5 | 8 | prospective target 与 trajectory audit |
| Clarity | 7 | 9 | 单公式、单故事优势 |
| Relevance | 7 | 8 | agent reliability 与一般 AI 方法连接 |
| Reproducibility | 7 | 9 | offline tasks、code、traces、hash |
| Overall | 4-5，reject | 7-8，accept-leaning | 由 sealed target 和 baseline victory 决定 |

## 13. Go / Pivot / Stop 门

### Gate 1：2026-07-14，72 小时机制门

继续必须同时满足：

- no-trigger invariance；
- 两个以上 patch 有足够 activation；
- 至少一个 patch 有跨模型 effect 或 rescue disagreement；
- 粗粒度 trigger conditioning 不比 raw source-copy 更差；
- 完整矩阵成本可承受。

未通过则不能以“再多跑一点也许会有”延长。

### Gate 2：2026-07-16，基础设施门

- 四个 endpoint 稳定；
- 36 tasks 可在同一 reference harness 运行；
- patch contracts 与 schema tests 全通过；
- source matrix 已开始，无系统性 telemetry 缺失。

否则停止 AAAI 强行冲刺，保留为后续 venue。

### Gate 3：2026-07-19，方法门

- source LOMO 中至少一个 target-aware 层级在 aggregate MAE 或 regret 上明确胜 source mean 与 nearest capability；
- 在 seal 前按最简充分原则冻结论文版本：若 failure-surface-nearest / opportunity-only 已足够，删除 typed response 主张；只有 TRACE-H 继续胜出时才保留 H-RESP；
- trigger strata 不依赖事后手工合并；
- abstract 能陈述已观察到的具体结果。

若所有 target-aware 层级都无增量，则当前 AAAI 主会方法线停止；不能因为已经写了 TRACE-H 就忽略更简单的结果。

### Gate 4：2026-07-20，Seal

冻结：

- target task-level predictions；
- 9 个 aggregate effect intervals；
- deploy/reject/abstain；
- scoring code、threshold、exclusion rules；
- commit hash 与时间戳。

### Gate 5：2026-07-23，Target audit

先生成不可修改的 prospective score report，再做任何 retrospective analysis。若 target 失败，禁止把事后重新拟合结果写成 prospective。

## 14. 最终决策

**接受 TRACE-H 作为当前唯一 AAAI-27 主线；条件继续 72 小时。**

理由：

1. 它显著修复了旧版最主要的黑盒、小样本和范围问题；
2. 它正面承接 SEAGym 的 H-FS 与 LIFE-HARNESS 的广泛迁移结果，并把尚未检验的 atomic prospective estimand 说清；
3. 它的实验可被明确杀死，不会无限延长；
4. 它把 17 天内的资源集中在一个真正决定审稿结果的 sealed-target claim 上。

同时必须对老师明确：

- 现在值得做的是 72 小时 pilot，不是承诺一定投稿；
- idea 质量约 8.2，不代表论文完成度高于 2；
- 当前最大风险仍是零项目证据和 17 天 deadline；
- 若 source LOMO 只支持简单 H-FS/H-OPP，就应主动简化；若所有 target-aware 方法都不胜 source mean，这个 idea 不应继续包装。

## 15. 本决策的后果

- 正式 Proposal 改为 [TRACE-H](../proposals/trace-h-formal-proposal-zh.md)；
- 旧 FORECAST-H 通用 predictor 降为 baseline，不再是方法中心；
- SEAGym 定义为 H-FS 假说来源与强基线依据，LIFE-HARNESS 定义为广泛迁移的正面先验；不得再写成“碰撞后回避”；
- intervention 限于三个 dormant event-gated patches；
- 形式化动作改为 deploy/reject/abstain；
- 删除当前阶段的 universal risk-control claim；
- Harness-Bench 作为候选主任务底座，ToolBench-X 仅为相关工作/未来 audit；
- 新建中文 72 小时 protocol，并在 pilot 后生成 DR-0003；
- 此后面向老师的研究文档默认中文。

## 变更记录

- 2026-07-11：基于新增 10 篇方法/碰撞论文和本地代码可用性审计，创建 DR-0002，替代 FORECAST-H 方法中心与旧评分。
- 2026-07-11：逐页复核 SEAGym、LIFE-HARNESS 与九篇近邻全文；修正此前“SEAGym 已占完整机制”的过度判断，将 novelty 从 7.3 调整为 7.8、机制证据从 7.2 调整为 7.8、idea 总评从 8.0 调整为 8.2；接受概率不因无新数据而上调。
