# 文献纳入与排除规范

- **初始本地快照截止：** 2026-07-10
- **增量检索开始：** 2026-07-11，Asia/Shanghai
- **当前状态：** 未饱和，提交前必须继续检索

## 直接证据纳入条件

论文至少研究以下一项：

- harness、scaffold、runtime policy、prompt、memory、skill、tool interface、verifier、retry 或 context policy 的跨模型/跨 backend 迁移；
- harness intervention 的 held-out-model evaluation；
- model-specific 与 model-invariant harness effects；
- model-harness cell 的 ranking reversal 或 effect heterogeneity；
- 固定 runtime infrastructure 下的 model replacement / model upgrade；
- unseen model-scaffold 或 model-intervention outcome prediction；
- harness configuration 的 warm-start 或 transfer-aware optimization；
- event-gated patch 的 activation、conditional recovery、harm 或 cost；
- target baseline behavior 到 intervention effect 的 prospective prediction。

## 方法证据纳入条件

- causal transportability 与 treatment-effect generalization；
- standardization、hierarchical effect prediction 与 meta-analysis；
- negative-transfer-aware multi-task / transfer optimization；
- prospective validation 与 sealed-target evaluation；
- stochastic agent 的 paired experimental design；
- selective prediction、abstention、calibration 与 decision regret。

## 反证纳入条件

- 大范围正迁移；
- 明显负迁移或 sign reversal；
- base model 完全主导 harness effect；
- deterministic contract 能迁移而 prose / learned policy 失败；
- operating point 或 budget coupling 使简单 portability 失效；
- conditional rescue 在相同 trigger type 下仍强烈 model-specific。

## 排除出直接集合

- 一个新模型在多个 benchmark 上使用同一固定 harness；
- 单模型内普通 train/test generalization；
- 没有模型或 harness 变化的纯 cross-domain transfer；
- 只在参考文献中偶然出现 `cross-model transfer`；
- 无法辨认改变了什么机制的完整 agent comparison；
- 方法/实验细节不足，无法判断 transferred object 与 target protocol；
- 只有观点、博客或二手摘要且无可核查 primary source。

## 每篇直接论文必须记录

1. transferred object；
2. source model 与 task distribution；
3. target model 与 task distribution；
4. atomic 或 bundled intervention；
5. tools、budget、timeout 与 information 是否受控；
6. target 是普通 held-out 还是 prospective seal；
7. descriptive transfer 或 predictive transfer；
8. positive、null、negative 或 mixed effect；
9. uncertainty、任务数和重复数；
10. activation opportunity 与 conditional response 是否分开；
11. limitation 如何界定剩余空白。

## 检索饱和规则

单个 query 没有新标题不算饱和。至少需要：

- 三个 query families 连续没有新 direct paper；
- 每篇 Tier A direct paper 的 backward citation check；
- 按关键作者、方法名和 exact phrase 做 forward/latest search；
- title/abstract search 与本地 full-text search；
- 对 benchmark、代码仓库和 contemporaneous preprint 做单独检索；
- dated search log 记录正结果和负 query。

