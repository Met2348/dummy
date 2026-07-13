# TRACE-H Policy Transport Claim Register

| Claim | 所需证据 | 最强碰撞/基线 | Kill condition | 当前状态 |
|---|---|---|---|---|
| Harness 是可学习 control policy | Harness MDP/action schema 与实际 controller | Offline-RL Harness | 不作为 TRACE-H 首创 claim | 文献支持；非本项目贡献 |
| 同一 harness action response 随 executor 改变 | same-prefix source branch effects + model interaction | MASA；SkVM；SEAGym | 无可重复 cross-executor heterogeneity | 文献支持；项目未验证 |
| Counterfactual branch supervision 优于普通 offline trajectory learning | Branch-AW vs ordinary Source-AW 的 source LOMO 与 target utility | Offline-RL Harness；behavior cloning | branch 增量只在 diagnostics 或持平 | 未验证 |
| Target baseline state support 可帮助 policy transport | baseline-only calibration + held-out executor execution | Metric Freedom；Nearest-AW；Category Router | MF/nearest/category 已足够 | 未验证 |
| Response-aware state metric 优于 semantic/task similarity | semantic-only、structured-only、response-aware ablation | SkillRouter；Adaptive Auto-Harness | end-to-end utility 无增量 | 未验证 |
| Partial transport 避免 target-private state 负迁移 | partial OT vs balanced OT/kNN，matched-mass risk curve | UniOT；selective prediction | balanced OT/kNN 持平或更强 | 未验证 |
| Conservative LCB/NONE 改善 target policy utility | posterior-mean argmax vs LCB policy | Bayesian-Agent；MESA-S | 只降低 coverage，utility/regret 不改善 | 未验证 |
| TRACE-H 在零 target action feedback 下运输 executable policy | whole-executor seal + router artifact + target ledger | MASA DS-Adapter；Source-AW；MF-Gated AW | 不胜 strongest same-budget baseline | 未验证 |
| TRACE-H 提高 target end-to-end success/utility | ALFWorld/WebShop final-test primary table | MASA；SkillAdaptor；SkVM-style；Offline-RL Harness | 只改善 prediction/process metric | 未验证 |
| TRACE-H 降低达到目标效用所需的 target feedback | 0/5/10/20/40 action-outcome frontier | SkillAdaptor；target-AW；MASA evolution | 少量 target feedback 方法全面更优且成本很低 | 未验证 |
| Transport-regret 可由 source error、alignment cost、unmatched mass 分解 | theorem/proposition + assumptions + empirical term audit | PAR；OT policy adaptation | 只能给无关或不可检验 bound | 未验证 |
| Router distillation 保留 utility 且降低开销 | OT router vs distilled router | MASA-Rewriter；lightweight controllers | fidelity/utility 显著下降 | 次要、未验证 |

## 状态规则

- `source-supported` 不能升级为 target-supported。
- 需要 target execution 的 claim 只有在 policy seal 后完成 primary table才能升级。
- effect MAE、OT distance、trigger frequency、HMS 或 cluster visualization 均是 secondary，不得升级 end-to-end claim。
- strongest baseline 指同 target-information budget 下预声明集合中的最强者，不能事后挑选。
- MASA evolved、SkillAdaptor、target-AW 等 extra-feedback 方法必须单列 target outcomes，不混入 zero-feedback 主表。
- 作者 artifact、作者 code、修复版和机制重实现必须分别标注。
- 如果 kNN/Source-AW 与 TRACE-H 持平，必须删除无增量组件或停止方法 claim。
- 如果 cross-family target 发生实质负迁移，不得用 in-family aggregate 掩盖。
- 任何 claim 改写必须有 decision record；当前 method center 为 [DR-0003B](../decisions/0003b-trace-h-policy-transport-pivot-zh.md)。

## 明确禁止的首创表述

- 首个 learned harness controller；
- 首个 model-aware skill adaptation；
- 首个 skill compiler/runtime；
- 首个 dynamic harness router；
- 首个 baseline-only skill utility predictor；
- 首个 target trajectory skill adaptation。

当前只允许争取：target-intervention-free、event-level response、whole-executor holdout、executable policy transport 的严格方法贡献。
