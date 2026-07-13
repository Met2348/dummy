# DR-0003B：TRACE-H 从效应诊断转为 Harness Policy Transport

- **日期：** 2026-07-11
- **状态：** 当前 method-center 决策
- **取代：** DR-0003 中的 patch-effect predictor method center
- **保留：** DR-0003 的文献碰撞、零自有证据、deadline 与 seal 风险判断
- **下一决策：** 72 小时 pilot 后建立 DR-0004

## 1. 决策

停止把 TRACE-H 设计为“预测多个 patch 在目标模型上的 aggregate effect”。批准新的方法中心：

> 从来源 executor 的 event-level counterfactual branch responses，向完全未见且无 action feedback 的目标 executor 运输一个 state-conditioned executable harness policy。

方法采用三项耦合机制：

1. **Counterfactual Branch Bank：** 同一来源 trajectory prefix 分支不同 harness actions，获得 action advantage；
2. **Response-Aware Partial Transport：** 用目标 baseline states 对齐来源 response states，允许 target-private state 不匹配；
3. **Conservative Policy Compilation：** 以 transported lower confidence bound 编译 action/NONE runtime router。

正式方案见[Policy Transport Proposal](../proposals/trace-h-policy-transport-proposal-zh.md)。

## 2. 为什么旧方向不够

旧方向的主要产物是：

- patch effect prediction；
- signed MAE/calibration；
- choose patch/none；
- prospective seal。

即使协议严谨，评审仍可能把它视为 evaluation/diagnosis，因为它不必改变 agent 的运行过程，第一主结果也不是 task performance。扩张后的 design paper 必须输出一个可运行 policy，并在共同 benchmark 上直接提高 success/utility。

## 3. 扩张后的直接碰撞

本轮全文审计确认：

- Offline-RL Harness 已提出 learned Harness MDP controller；
- MASA 已提出 target-backbone-conditioned skill evolution/rewriter；
- SkVM 已提出跨 model+harness 的 skill compiler/runtime；
- Adaptive Auto-Harness 已提出 harness tree 与 solve-time routing；
- SkillAdaptor 已提出 target failure attribution、修改和 qualification；
- Metric Freedom 已提出 baseline-only generic headroom。

因此，新颖性不能来自 controller、router、compiler、model-aware 或 baseline-only 中任一单词。剩余方法空间是它们尚未共同覆盖的：

```text
whole executor held out
+ zero target harness-action outcomes
+ event-level action response transport
+ executable runtime policy
+ end-to-end method PK
```

证据审计见[设计方法扩张全文边界](../foundations/notes/method-design-evidence-boundary-zh.md)。

## 4. 方法主张

### 大 claim

Harness control policy 可以在基础模型 dynamics 变化时，通过 target baseline state support 和 source event responses 进行运输，而不必先在目标模型上搜索每个控制动作。

### 核心机制 claim

对 target-private states 保留 unmatched mass，并基于 response uncertainty 选择 NONE，可比 source policy copy、kNN 或 forced balanced alignment 减少负迁移。

### 实证 claim

在整个目标 executor 留出的 ALFWorld/WebShop tests 上，TRACE-H 的 actual task utility 超过最强 same-target-information baseline。

### 不再作为主 claim

- generic skill utility prediction；
- aggregate patch effect MAE；
- harness interaction diagnosis；
- patch selection benchmark；
- 单纯的 sealed protocol。

## 5. 必须 PK 的方法

### 相同目标信息预算

`No Harness, best fixed, Source-AW, Nearest-AW, MF-Gated AW, Category Router, kNN-Branch, Balanced-OT, PAR-style penalty`。

### 直接论文系统

`MASA Base/DS/Evolved, SkillAdaptor, target-trained Offline-RL Harness, SkVM-style AOT, Partial Harnessing`。

三类信息预算分别记录：source episodes、target baseline episodes、target action outcomes。完整协议见[方法 PK 矩阵](../notes/trace-h-method-pk-matrix-zh.md)。

## 6. 主指标重排

### Primary

1. target success rate；
2. normalized end-to-end utility；
3. relative gain over strongest same-budget method；
4. target policy regret；
5. success-cost Pareto frontier。

### Secondary

action-effect error、alignment cost、calibration、matched mass、state cluster 和 trigger frequency。

只改善 Secondary 而不改善 Primary，项目判定失败。

## 7. 评分变化

| 维度 | Policy Transport 版 | 理由 |
|---|---:|---|
| Neatness | 8.5/10 | branch、transport、compile 三段可用一张图解释 |
| Excitement | 8.9/10 | 从预测工具升级为模型更换时的零反馈控制策略迁移 |
| Problem evidence | 9.6/10 | 多篇方法共同证明 controller、model mismatch 与 target adaptation 重要 |
| Mechanism evidence | 8.4/10 | branch support、dynamics adaptation、partial alignment 均有基础，但组合未验证 |
| Novelty | 7.8/10 | direct components 均有先例；cross-executor zero-feedback policy transport 仍有空间 |
| Technical depth | 8.8/10 | Harness MDP、branch counterfactual、partial OT、regret decomposition |
| Soundness ceiling | 8.7/10 | whole-executor seal、actual execution、强 baseline |
| Workload feasibility | 3.9/10 | 155-230 小时，baseline 复现和分支 runner 很重 |
| Community fit | 8.8/10 | agent learning、domain adaptation、systems control 三方均可读 |
| Current own evidence | 1.0/10 | 尚未运行 pilot |
| Idea overall | 8.1/10 | 方法形态更强，但工程与碰撞风险同步上升 |

评分提高不代表投稿概率已提高。无自有 end-to-end result 时，当前无条件 AAAI 接受概率仍只按 **3-7%** 规划；若 pseudo-target 明显胜最强同预算 baseline，可升至 15-25%；若两个 sealed targets 含 cross-family 均成立，可按 28-42% 规划。这是资源决策估计，不是统计概率。

## 8. 工作量与资源

- 预计 `155-230` 聚焦人时；
- pilot 约 `750-900` runs；
- 需要 branch/replay、state hash、统一 actions、三个公开 baseline、partial OT 和 append-only ledger；
- 当前 RTX 5090 Laptop 24GB 可支持 4B-14B 本地量化实验，32B target 必须先做显存/吞吐 smoke test。

相比诊断版，工作量更大但论文产物更像方法。若 72 小时内 runner/branch support 不稳定，必须停止完整矩阵，不得用诊断结果替代。

## 9. Go/Pivot/Stop

### Go

TRACE-H 在 pseudo-target 的 end-to-end utility 上胜 Source-AW、MF-Gated、kNN 与 balanced OT 中最强者，且 partial transport/LCB 至少一项有实际 utility 增量。

### Pivot

- kNN 与 TRACE-H 持平：删除 OT，保留 branch retrieval router；
- source AW 持平：转 cross-executor benchmark，不投新方法；
- 少量 target feedback 全面占优：转 target-feedback sample-efficiency 方法；
- MASA DS-Adapter 零 action feedback 更强：研究 static rewrite + runtime control 的互补，而非声称替代。

### Stop

- 只改善 diagnostics；
- cross-family target 负迁移；
- branch outcomes 不可重复；
- strongest same-budget baseline 持平或更强；
- target action outcome 泄漏。

## 10. 最终表述

> TRACE-H 不是判断 harness 是否值得用的诊断器，而是一个把来源模型上的控制响应运输成未见目标模型 runtime policy 的设计方法。它必须靠目标任务效用击败 learned controllers、model-aware adapters 和简单 transport baselines，而不是靠更窄的文献空白成立。
