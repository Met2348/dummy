# Idea 大调整记录：从 FORECAST-H 到 TRACE-H

> **历史提示：** 本文记录第一轮从 FORECAST-H 到 TRACE-H 的选择。第二轮全文检索发现 Metric Freedom 已占据 broad baseline-only skill-utility prediction；当前评分与实验门以 [DR-0003](../decisions/0003-metric-freedom-collision-reassessment-zh.md) 为准，本文不做事后改写。

- **日期：** 2026-07-11
- **目的：** 记录正式 Proposal 形成前的候选比较，避免结果出来后倒推研究动机。
- **结论：** 选择 V2“触发条件化标准化”，放弃通用黑盒预测与当前数据不支持的形式化保证。

## 1. 新增检索改变了什么

本轮在原 68 篇隔离语料之外补读并下载了 10 篇近邻或方法论文，又对 SEAGym、LIFE-HARNESS 及九篇最接近工作逐页复核正文、附录和 limitations。最重要的新信息是：

1. [Harness-Bench](../foundations/papers/2605.27922-harness-bench.pdf) 已经完成大规模 model-harness configuration 测量，另做完整配置 heatmap 不再新颖。
2. [SEAGym](../papers/2606.17546.pdf) 已观察到跨 backend 迁移的非对称与符号反转，并通过 update artifact case study 提出 failure-surface alignment explanation；但它没有定量 surface、atomic effect、target outcome seal 或 predictor。H-FS 思想不属于我们，前瞻检验仍是空白。
3. [LIFE-HARNESS](../papers/2605.22166.pdf) 从 Qwen3-4B 演化的环境专用整套 harness 可复用于 17 个其他 backbone，126 个单元中改善 116 个；这说明不能只讲负迁移，TRACE-H 必须在高正迁移先验下识别少数例外。
4. [HASP](../foundations/papers/2605.17734-hasp.pdf) 已发明带 `should_activate` / `intervene` 接口的事件触发 Program Function，不能再把“可执行触发 patch”当贡献。
5. [ContractSkill](../foundations/papers/2603.20340-contractskill.pdf) 已经用显式 contract、deterministic verifier 和 local repair 证明一定范围的 artifact portability。
6. [ToolBench-X](../foundations/papers/2606.25819-toolbench-x.pdf) 已在多个模型上比较异常环境、恢复提示与额外推理；通用“错误提示有用”不新颖。
7. [The Verifier Tax](../foundations/papers/2603.19328-verifier-tax.pdf) 已测量 verifier intervention frequency、recovery 和 cost trade-off。
8. [Causal transportability](../foundations/papers/2012-transportability-completeness.pdf) 与 [target-population generalization](../foundations/papers/2019-generalizing-trials.pdf) 都要求明确的结构差异或可识别性条件；本项目不应把普通跨模型回归包装成一般因果迁移。
9. [Conformal Risk Control](../foundations/papers/2024-conformal-risk-control.pdf) 要求 exchangeability 和 monotone loss；四个模型的 sparse cells 不支持无条件风险保证。

这些论文没有终结“预测密封目标上的 patch effect”。它们迫使我们把贡献从 artifact、benchmark 和通用 predictor 收缩到一个可检验的机制规律，同时要求正面解释“广泛迁移为何成立、少数例外为何发生”。详细证据见[全文碰撞复核](../foundations/notes/fulltext-collision-reassessment-zh.md)。

## 2. 候选版本

### V0：完整 Harness 迁移热图

**问题：** 比较多个模型、多个完整 agent harness，找 ranking reversal。

**优点：** 数据直观，容易解释。

**淘汰理由：** Harness-Bench、Stop Comparing LLM Agents、SEAGym 等已占据。再做一张矩阵最多是复现或 benchmark extension。

### V1：FORECAST-H 通用目标效应预测

**问题：** 用模型、任务、预算、intervention descriptors 和 baseline target fingerprints 学习 `Delta(target, patch)`。

**优点：** 与 absolute performance prediction 有明确区别；prospective seal 有可信度。

**缺点：** 特征多、模型少、机制弱。即使 ridge 胜出，审稿人也可能认为是一个数据集上的经验拟合；复杂 hierarchical/Bayesian 模型会显得装饰性大于必要性。

### V2：TRACE-H 触发条件化标准化

**问题：** 对 dormant-until-trigger patch，用目标 baseline 失效机会分布与来源 conditional rescue/harm profile 预测目标效应。

**核心式：**

```text
Delta(target, patch)
  = sum_k P_target(first_trigger=k)
          * E_source[paired_delta | first_trigger=k]
```

**优点：**

- 方程由 patch 执行语义和全期望公式导出；
- target feature 只有与 patch 直接相关的 failure surface；
- 条件稳定性是明确、可被反驳的科学假设；
- 方法可以很简单，减少小样本过拟合；
- 与 HASP/ContractSkill 的关系是使用其已建立的 artifact class，而非重复发明；
- 与 Harness-Bench 的关系是从 configuration-level measurement 前进到 atomic prospective effect prediction；
- 与 SEAGym 的关系不是避开 failure surface，而是把其 H-FS 解释变成 baseline-only target、patch-outcome-sealed 的定量检验，并接受针对性基线挑战；
- 与 LIFE-HARNESS 的关系是解释其高迁移先验下为何仍存在持平/下降单元，而不是否认整套 harness 可复用。

**主要风险：** 条件 rescue 仍可能强烈 model-specific；即使有效，也可能不胜“选择 failure surface 最相似来源并复制其效应”的简单方法。后者若在 sealed target 上成立，应简化成 H-FS/H-OPP 前瞻验证；只有所有 target-aware 方法都无增量时，当前方向才真正失败。

### V3：Conformal / distribution-free 安全部署

**问题：** 对目标 patch deployment 给出有限样本负迁移概率保证。

**优点：** 理论外观强，容易形成 risk-control 叙事。

**淘汰理由：** calibration unit、exchangeability、monotonicity 和样本数都不充分。现在加入保证会降低 soundness，而非提升 technical depth。

### V4：复用 Harness-Bench 公开结果做最佳 Harness 选择

**问题：** 在 5,194 条已有轨迹上预测 unseen model 的最佳完整 harness。

**优点：** 数据量大，运行成本低。

**淘汰理由：** 完整 harness 同时改变工具、状态、权限、记忆和执行逻辑，无法归因到 patch mechanism；结果更接近 algorithm selection，无法支撑原子迁移主张。公开仓库当前也没有附带 5,194 条完整轨迹。

## 3. AAAI 维度候选比较

以下为实验前的主观 idea ceiling，不是论文结果评分。

| 版本 | Neat | Exciting | Novel | Sound | Feasible | AAAI fit | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| V0 完整配置热图 | 6.0 | 6.2 | 3.5 | 7.0 | 6.5 | 5.5 | 已占据 |
| V1 通用效应预测 | 7.2 | 7.8 | 7.0 | 6.3 | 3.8 | 7.5 | 太宽、太黑盒 |
| V2 TRACE-H | 8.7 | 8.3 | 7.8 | 8.1 | 5.8 | 8.3 | **选择** |
| V3 形式化风险保证 | 8.0 | 8.5 | 8.0 | 4.5 | 2.5 | 7.5 | 假设不成立 |
| V4 完整 Harness 选择 | 6.8 | 7.0 | 5.0 | 6.2 | 7.0 | 6.3 | 可作后续副项目 |

V2 的技术深度来自 estimand、contract、条件标准化、uncertainty 和 prospective audit 的一致性，而不是模型复杂度。

## 4. V2 的最小可发表中心

正文必须始终围绕一个问题：

> baseline failure surface 能否预测一个事件触发 patch 在未见模型上的 paired effect？

最小主会故事只包含：

1. 三个有 dormant contract 的 patch；
2. 三个来源模型与一个密封目标；
3. 一个 task suite 和 36 个可执行任务；
4. 一个 trigger-standardization estimator；
5. 一组递进 baselines：capability-nearest、failure-surface-nearest、opportunity-only，再到 typed conditional response；
6. 一个 prospective target audit；
7. 一个 deploy/abstain regret 结果。

第二 benchmark、完整 harness 比较、自动 patch 生成、复杂表征学习和严格 conformal guarantee 均不进入主线。

## 5. 新颖性防线

面对审稿人，正确回答应是：

- **对 Harness-Bench：** 它测量完整配置；我们固定 harness，估计原子 patch 的 paired delta，并预测未见模型。
- **对 SEAGym：** 它事后观察 bundled harness 的跨 backend 迁移并用 artifact case study 支持 H-FS；我们正面预注册 H-FS/H-OPP/H-RESP，测量目标 baseline 失效面，对原子 patch 生成有符号效应预测并密封 outcome。只有 H-RESP 胜过 failure-surface-nearest / opportunity-only 时才主张完整 TRACE-H。
- **对 LIFE-HARNESS：** 它证明环境专用 bundled harness 广泛跨模型迁移；我们不重复整套迁移矩阵，而研究在 deployment 前能否识别其未改善或有害的局部例外，并分解 opportunity 与 response。
- **对 HASP：** 它构造并使用 PF；我们不发明 PF，而预测一个既有 PF/patch 换模型后的 effect。
- **对 ContractSkill：** 它证明 matched-layer portability；我们预测 target effect sign/magnitude 并允许 abstain。
- **对 ToolBench-X：** 它诊断 hazards 并比较 hints；我们研究 baseline failure distribution 是否能把 source conditional response 标准化到 sealed target。
- **对 PACE：** 它预测绝对 agent score；我们预测相对同一 baseline 的 intervention effect。
- **对 selective prediction：** abstention 是已有原则；贡献是它在 prospective harness-effect transport 中的操作化和评估，不是 abstention 本身。

## 6. 最危险的反驳

### “这只是全期望公式”

公式本身当然不是新数学，failure-surface 解释也不是本文首次提出。论文价值必须来自三个部分共同成立：

1. dormant contract 让 no-trigger effect 可审计地为零；
2. failure strata 使 conditional response 比 aggregate effect 更稳定；
3. 该规律在真正密封的新模型上改善预测和部署；若只胜 source-copy 而不胜 failure-surface-nearest / opportunity-only，则收缩为 H-FS/H-OPP 验证，不保留完整 H-RESP 主张。

若只有第一项，论文不够；若所有 target-aware 方法都没有第三项，不能冲主会。简单 H-FS/H-OPP 若在 sealed target 上强成立，可以成为更克制的替代版本，而不是因为与近邻接近就自动丢弃。

### “baseline 跑了同一批 target tasks，不算 unseen”

本文是 baseline-observed、intervention-outcome-sealed 的 prospective transductive setting，而不是 zero-interaction transfer。必须在标题、摘要和 protocol 中明确。附加 disjoint-task audit 有资源时再做，不能偷换主设定。

### “四个模型太少”

降低主张范围：不是发现普适 model law，而是测试一个 mechanism-guided estimator。使用 task-level paired outcomes、source leave-one-model-out 和单独的 prospective target；仍需把模型数量限制列为主要外部有效性风险。

### “patch 是手工挑的”

patch 选择依据 observable runtime boundary 和已被 HASP、ContractSkill、Verifier Tax 等文献反复使用的机制，不依据 pilot 结果。发布 contract 与实现，并把 patch family 的一般性作为限制而非过度声称。

## 7. 最终选择

**选择 V2 TRACE-H，状态为 conditional-go。**

该选择只提高了 idea 的审稿潜力，没有提高当前 submission readiness。逐页全文复核后，SEAGym 应被视为 H-FS 的假说来源而非 exact method collision，LIFE-HARNESS 则提供广泛迁移的强正面先验；因此新颖性校正为 7.8，而不是此前过度保守的 7.3。当前项目自己的 empirical evidence 仍为零；72 小时 pilot 是从“好 idea”进入“可能的论文”的唯一通道。
