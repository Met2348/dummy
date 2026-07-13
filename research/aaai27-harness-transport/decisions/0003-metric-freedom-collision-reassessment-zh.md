# DR-0003：Metric Freedom 碰撞后的 TRACE-H 再评估

> **方法中心已更新：** [DR-0003B](0003b-trace-h-policy-transport-pivot-zh.md)已将项目从诊断型 patch effect prediction 转为 cross-executor executable harness policy transport。本文的 Metric Freedom 碰撞、概率和证据风险仍有效，但当前算法、baseline PK 和成功标准以 DR-0003B 与[新 Proposal](../proposals/trace-h-policy-transport-proposal-zh.md)为准。

- **日期：** 2026-07-11
- **状态：** collision-aware conditional-go
- **取代：** DR-0002 的 novelty、总评、接受概率与实验门
- **保留：** DR-0002 的 deadline、工程事实、dormant contract 与 prospective seal 纪律
- **证据入口：** [第二轮全文碰撞审计](../foundations/notes/second-wave-fulltext-collision-audit-zh.md)
- **下一次决策：** 72 小时 pilot 后建立 DR-0004

## 结论摘要

第二轮全文检索发现了一个必须正面承认的直接碰撞：`From Multi-Agent to Single-Agent: When Is Skill Distillation Beneficial?` 已提出 Metric Freedom，用目标模型无 skill 的 baseline runs 在任何 skill 执行前预测 skill utility，并在两个 backbone 上复核趋势。

因此，TRACE-H 不能再声称：

> 首次只用目标 baseline 行为，事前预测 harness/skill intervention 是否有用。

这不是措辞问题，而是问题层贡献已经被占据。若继续，主问题必须收窄为：

> 给定多个冻结 dormant atomic patches，能否在未见目标模型上分别预测相对同一 baseline 的 signed effect 与 interval，并在 Metric Freedom-informed baseline 之上改善 patch choice、negative transfer 和 deployment regret？

这一剩余问题仍有价值：Metric Freedom 是 metric-level generic headroom signal，无法区分同一目标上的不同 patch；其主预测证据是 13 个聚合点相关性，没有封存的 task/model prediction set、效应校准或部署 regret。但剩余新颖性明显低于第一轮判断，且需要更多 baseline repeats 和更强对照。

当前决策：**仍可投入一次 72 小时 collision-aware kill test，但不再把 TRACE-H 视为高把握 AAAI 主线。只有它在 pseudo-target 上明显胜过 Metric Freedom + source prior，才继续完整矩阵。**

## 1. 什么事实改变了决策

### 1.1 Metric Freedom 的全文事实

- `F = 1-r_M`，其中 `r_M` 是 baseline 行为距离与 score-distance 的 Mantel/Spearman 一致性；
- 每个问题 `n=6` 次无 skill runs，每个数据集 `m=6` 个问题；
- `F` 在任何 skill 执行前计算；
- 主分析覆盖 4 tasks、11 datasets、6 metrics、13 个有效聚合点；
- Sonnet 4.6 报告 `r=-0.85`，GPT-5.1 用相同 skills/prompts 复核 `r=-0.71`；
- 论文明确将其定位为 a priori predictor of skill utility。

所以“baseline-only prospective utility”不能再归本项目。

### 1.2 仍然不同的部分

- Metric Freedom 每个 task/metric 只有一个 `F`，对多个候选 patch 不具辨别力；
- 它研究 bundled MAS-derived skill，不是 dormant atomic patch；
- 它报告相关性与 headroom-normalized lift，不报告 patch-specific signed MAE、Brier、interval coverage 或 regret；
- 13 个点用于总体相关性，没有先冻结映射再在未见 task/model 上评分；
- GPT-5.1 复核仍使用同一 tasks、metrics 与 skill definitions。

这些差异足以形成实验问题，不足以凭文字自动形成主会 novelty。

## 2. 修订后的论文中心

旧中心：

```text
target baseline failure surface
  x source conditional response
  -> target patch effect
```

修订中心：

```text
generic headroom controls
  = baseline strength + Metric Freedom

patch-specific forecast
  = target patch-addressable opportunities
  x source patch-specific rescue/harm

decision
  = choose none / choose one patch / abstain
```

论文必须回答的不是“baseline traces 有没有用”，而是：

1. Metric Freedom 与 baseline strength 是否已经足以决定 generic skill benefit？
2. 在相同 generic headroom 下，patch-specific opportunity 是否区分不同 patch？
3. typed conditional response 是否再提供来源可迁移信息？
4. 这些增量是否在 sealed target 上降低选错 patch 的 regret？

## 3. 评分总表

评分区分当前 idea、成功结果后的上限和当前准备度。

| 维度 | 当前 idea | 成功结果上限 | 置信度 | 第二轮判断 |
|---|---:|---:|---|---|
| Neatness | 8.4/10 | 9.0/10 | 中高 | patch-choice 仍可一图讲清，但多一个必须正面控制的 generic headroom 层 |
| Excitement | 8.3/10 | 9.0/10 | 中 | 若能显示同一 `F` 下不同 patch 反号并提前选对，会很有冲击力 |
| 问题证据 | 9.5/10 | 9.5/10 | 高 | 19 configs、6-model harness matrix、负迁移与非单调 benefit 证据充分 |
| 机制合理性证据 | 8.5/10 | 9.0/10 | 中高 | Metric Freedom、Harness Effect、Harness Updating 直接提供 headroom/adherence 机制 |
| 本项目方案证据 | 1.0/10 | 8.5/10 | 高 | 仍无一条自己的 trajectory |
| Novelty | 7.1/10 | 8.2/10 | 中 | 宽泛 predictor 已被占；只剩 patch-specific signed/calibrated/sealed decision |
| Technical depth | 8.0/10 | 8.7/10 | 中 | estimand、contract、标准化、校准与 decision evaluation 仍完整 |
| Soundness potential | 8.2/10 | 9.0/10 | 中高 | 强基线与 seal 提升可信度，但模型数仍少 |
| 工作量可行性 | 4.8/10 | 6.5/10 | 中低 | Metric Freedom 需要重复 baseline probes，最小矩阵从 576 升至约 936 episodes |
| AAAI community fit | 8.2/10 | 8.8/10 | 中 | 与最新 skill evaluation/selection 直接对话 |
| Clarity potential | 8.4/10 | 9.0/10 | 中高 | 必须压住 related-work 和 baseline 数量 |
| Reproducibility potential | 8.9/10 | 9.4/10 | 高 | 离线 tasks、contracts、seal 与 raw traces |
| 当前投稿准备度 | 2.0/10 | 8.0/10 | 高 | 文献边界更准，但结果仍为零 |

**当前 idea 总评：约 7.8/10，低于第一轮的 8.2。**

下调不是因为看到碰撞就回避，而是因为 broad novelty 已经实质减少。问题证据和机制证据反而上升。

## 4. Neatness

### 保住的 neat 部分

- 一个固定 reference harness；
- 三个可执行原子 patch；
- 一个 dormant contract；
- 一个 target outcome seal；
- 一个 patch-choice regret；
- 一条从 generic headroom 到 patch-specific response 的增量链。

### 新增复杂性

- Metric Freedom 需要 repeated baseline runs；
- 必须同时解释 score topology 与 failure opportunity；
- baseline family 增加，七页 related work 压力变大；
- 若三个 patch 的 `F` 都相同，必须把“F 是 generic control、TRACE-H 是 patch-specific”讲得非常清楚。

### 强制简化

- `F_out` 为主，`F_trace` 只作附录或 sensitivity；
- success delta 为主，cost utility 为次；
- 不加入 learned model embedding；
- 不做自动 patch generation；
- 只保留能区分最强基线的最小 trigger taxonomy。

## 5. Excitement

真正令人兴奋的结果不再是“baseline traces 能预测”，而是以下任一项：

1. 同一 target category 的 Metric Freedom 基本相同，但 P1/P2/P3 中至少一个有害、一个有益，TRACE-H 在 unseal 前选对；
2. baseline strength 与 `F` 都建议“skill 有 headroom”，但某个 patch 因 low opportunity 或 harmful conditional response 应被拒绝；
3. source mean 与 `MF-nearest` 预测错误符号，TRACE-H 正确并降低 top-1 patch regret；
4. 明确分离 generic metric rigidity、patch opportunity 与 post-trigger recoverability 三种机制。

兴奋度会在以下情况坍塌：

- `F + source mean` 已解释全部结果；
- 所有 patch 均为正且排序不重要；
- 只有 baseline success rate 有效；
- patch-specific conditioning 只增加方差；
- target seal 只有一个很小 aggregate cell，无法区分偶然性。

## 6. 证据强度

### 问题存在：A

- `A Framework for Evaluating Agentic Skills at Scale` 在约 500 skills、1,000 tasks、19 agent-model configurations 上直接测量 with/without-skill delta，模型 gain 从 5.5 到 22 分不等；
- `The Harness Effect` 在同一 frozen loop 的 6-model matrix 中得到 30 improve、11 flat、7 regress；
- `Harness Updating Is Not Harness Benefit` 显示 harness benefit 对 base capability 非单调，并定位 activation/adherence failure；
- MemDelta、Natural-Language Agent Harnesses、SkillCraft、SkillsBench 与 More Skills, Worse Agents 提供直接负迁移和符号异质性。

问题证据比第一轮更强。

### 机制合理：B+

- Metric Freedom 支持 generic score-topology/headroom；
- Harness Effect 的 `r=.99,n=6` 支持 baseline strength 强控制；
- Harness Updating 支持 activation/adherence 而非简单规模；
- SEAGym 支持 failure-surface alignment；
- AHE 的 regression blindness 表明“写出风险预测”远未等于准确预测。

但没有文献直接证明 source patch-specific conditional response 能迁移到 sealed target；该核心仍未验证。

### 本项目解法：F

仍然没有 contract test、pilot、LOMO、MF baseline、seal 或 regret。文献不能替代。

## 7. Novelty

### 已被占据

- baseline-only a priori skill utility prediction；
- target baseline behavior 作为 skill headroom signal；
- cross-backbone frozen skill replication；
- prospective edit manifest；
- frozen selector；
- posterior-guided skill maintenance；
- dynamic selective intervention/no-op；
- frozen harness/skill/controller transfer；
- risk gate/certificate framing。

### 仅剩的 defendable claim

> 我们研究的不是 generic skill headroom，而是 unseen target 上多个 frozen atomic patches 的比较效应。方法在不观察 target patch outcomes 时输出 patch-specific signed magnitude 与 interval，并以真正封存的 patch selection regret 和 calibration 评价。

### 为什么是 7.1

- 贡献主要来自 estimand 与 protocol conjunction，不是全新理论对象；
- Metric Freedom 已抢先建立最接近的 baseline-only predictor framing；
- AHE/Self-Harness/What Should/Bayesian-Agent 分别占据 protocol pieces；
- 但未发现工作同时完成 atomic patch、unseen target model、signed effect、seal、calibration 与 decision regret；
- 如果结果只证明 failure frequency 有用，审稿人很可能视为 Metric Freedom/SEAGym 的自然延伸；
- 只有 decisive patch-specific result 才能把 novelty 拉回 8 分以上。

## 8. Technical Depth 与 Soundness

保留的技术链：

- task-level effect estimand；
- dormant/no-trigger contract；
- generic headroom control；
- opportunity-response decomposition；
- source leave-one-model-out；
- target model seal；
- empirical interval/calibration；
- patch-choice regret 与 abstention。

新增 soundness 风险：

1. Metric Freedom 的 36 baseline probes/category-model 可能带来高成本；
2. 四个模型仍不足以估计模型总体规律；
3. `F` 与 baseline success、failure opportunity 可能高度共线；
4. 三个 patch 的 effect cells 可能过少，ranking/regret 不稳定；
5. source-specific `rho` 可能不迁移；
6. target 同任务 baseline 可见，仍是 transductive audit；
7. provider 非确定性可能使 baseline topology 与 patch effect 都含噪。

对应纪律：只做低维预声明 baseline；不把模型当随机样本；报告单元级结果；seal 前冻结缩放与阈值；没有共享 seed 时使用预声明重复；禁止 universal law 和 distribution-free guarantee。

## 9. 工作量

### 修订后最小完整设计

- 3 source + 1 sealed target；
- 36 tasks，三个 category；
- 18 个预声明 MF probe tasks，每题 6 次 baseline；其余 18 tasks 每题 1 次 baseline；
- 每模型 126 baseline episodes + 108 patch episodes；
- 总计约 `4 x 234 = 936` 个 primary + diagnostic episodes；
- 3 次 source LOMO + 1 次 prospective audit。

预计聚焦人时从 80-130 上调为 **100-155 小时**，另加模型墙钟时间。主要新增成本是 baseline-repeat orchestration、Metric Freedom distances、强基线实现与更严格审计。

在只有 17 天且 runner/endpoint 尚未验证的情况下，可行性是本 idea 的第二大风险，仅次于零结果。

## 10. 社区接受概率

### 接受理由

- 与 2026 年最新 skill evaluation 和 harness transfer 文献直接对话；
- 不回避最强 predictor，而是做严格增量；
- patch-specific signed decision 比 generic correlation 更接近部署问题；
- seal、calibration 和 regret 提高可信度；
- 负结果也能界定 baseline-only transport 的极限。

### 拒绝理由

1. “Metric Freedom 已经做了核心问题，你只是细化。”
2. “四个模型和一个 target 不能支持跨模型结论。”
3. “三个手工 patch 代表性不足。”
4. “全期望分解是显然的，结果增量又小。”
5. “MF baseline 实现不忠实或 probe budget 不公平。”
6. “目标 baseline 与同任务可见，不是真正 zero-shot。”
7. “936 episodes 仍缺可靠重复与显著性。”
8. “只在一个 benchmark，effect ranking 可能是 task-suite artifact。”

### 条件概率

| 投稿状态 | 主观接受概率 | 判断 |
|---|---:|---|
| 当前 proposal、无结果 | 0-2% | 不是论文 |
| 只有 pilot 与 heterogeneity | 2-6% | 最新近邻已覆盖现象 |
| 完整矩阵但与 MF/source baselines 持平 | 2-6% | 主方法无增量 |
| source LOMO 胜基线，sealed target 混合 | 8-18% | 有方法信号，关键证据弱 |
| 一个真正 sealed target 明显胜 MF family，patch regret 降低 | 20-32% | 有可辨识贡献，仍受模型数和单 benchmark 限制 |
| 两个独立 sealed targets 或第二 task family 复现 | 28-42% | 证据显著增强，但当前 deadline 很难完成 |

用于当前资源决策的无条件接受概率下调为 **3-8%**。它综合工程完成、Metric Freedom 增量、seal 成功、写作与评审风险，不是统计推断。

## 11. Go / Pivot / Stop

### 72 小时 Go

必须同时满足：

1. dormant contract 与 no-trigger invariance 通过；
2. 至少两个 patch 有足够 opportunities；
3. 至少一个 patch 出现 model-dependent effect/response；
4. Metric Freedom baseline 能按忠实协议计算；
5. sealed pseudo-target 上，patch-specific TRACE-H 在至少一个主要指标上明确优于 `MF + source prior`、baseline strength 与 source mean；
6. 增量不是由一个 infrastructure failure 或事后 taxonomy 合并造成；
7. 936-episode 完整设计在成本和时间上可承受。

### Pivot

- **MF wins：** 转为 Metric Freedom 的独立复核/边界研究；当前 AAAI 主方法停止。
- **Opportunity only wins：** 收缩为 patch opportunity standardization，不保留 typed response。
- **One patch only：** 聚焦一个 patch family，扩大任务数和 target models。
- **Active probe wins：** 转向最少目标 intervention probes 的主动校准；放弃 baseline-only claim。
- **Process metrics win：** 若 activation/adherence 解释一切，转为 harness-following diagnostic，不声称 effect transport。

### Stop

- TRACE-H 与 `MF + source prior` 持平或更差；
- patch-specific effect 不存在或全部同号同序；
- target seal 泄漏；
- baseline repeats 无法稳定估计 `F`；
- 条件 strata 只增加方差；
- 到 2026-07-14 仍无稳定 runner/endpoint；
- 新论文完成 exact atomic sealed patch-effect forecast。

## 12. 最终决策

**保留 TRACE-H，但从“唯一推荐主线”降为“collision-aware conditional go”。**

理由：

1. Metric Freedom 是实质碰撞，必须下调 novelty 与概率；
2. patch-specific signed/calibrated decision 仍未找到 exact match；
3. 该差异可以在 72 小时内被强基线直接杀死；
4. 若它无法胜过 Metric Freedom-informed baseline，就不值得为 AAAI deadline 继续投入；
5. 若它能在 seal 前选对不同符号 patch，反而会形成比原 idea 更具体、更可信的贡献。

对导师的准确表述应是：

> 我们找到了一篇真正占据 broad claim 的论文，所以不再说“首次预测 skill 是否有用”。现在只检验更难的剩余问题：能否在新模型上区分多个原子 patch 的正负与最优选择，并明显胜过 Metric Freedom。这个问题仍有价值，但 idea 总评从 8.2 下调到 7.8，当前无条件 AAAI 接受概率从 5-12% 下调到 3-8%。

## 13. 决策后果

- 正式 Proposal 必须加入 Metric Freedom 第一强基线与相关工作；
- 主指标加入 top-1 patch regret、choose-none 和 calibration；
- pilot 规模从 144 调整为 234 episodes；
- 完整矩阵预算约 936 episodes；
- 原计划的 pilot decision 编号顺延为 DR-0004；
- DR-0002 保留为第一轮历史快照，不事后改写其评分；
- 提交前必须继续监控 `Metric Freedom`、`skill utility prediction`、`harness benefit prediction` 关键词。
