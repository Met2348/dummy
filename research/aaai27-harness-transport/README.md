# AAAI-27 Cross-Executor Harness Policy Transport 研究项目

本目录与宽泛语料库 `research/aaai27-harness-frontier/` 隔离，专门用于检索、证伪和执行 TRACE-H 方向。

## 当前研究主张

TRACE-H 研究 **Cross-Executor Harness Policy Transport**：在完全不读取目标模型任何 harness-action outcome 的条件下，从来源模型的 same-prefix action branches 学习事件级 response，把目标 baseline states 部分对齐到这些 response states，并编译出在运行时选择 `CHECK/RETRY/REPLAN/NONE` 的可执行 controller。主贡献必须体现在未见 executor 的实际 success、cost-adjusted utility 和 policy regret，而不是 effect prediction 或 transport diagnostics。

项目不把以下内容单独视为贡献：

- 再做完整 model-harness comparison；
- 再证明 harness 会正迁移或负迁移；
- 发明 event-gated skill 或 contract artifact；
- 只报告 trigger frequency、recovery rate 或 heatmap；
- 只预测目标模型的绝对 agent score；
- 再声称首次从目标 baseline runs 事前预测 generic skill utility。
- 再声称首个 learned harness controller、model-aware skill adapter、skill compiler 或 dynamic router。

正式贡献必须包含 counterfactual branch bank、whole-executor holdout、target-action zero-feedback seal、response-aware partial transport、conservative runtime policy，以及与 Offline-RL Harness、MASA、SkillAdaptor、SkVM-style、Metric Freedom、kNN 和 balanced OT 的公平 PK。全文复核后的定位是：已有方法占据各个单独模块，TRACE-H 只能靠 cross-executor policy transport 的新算法与 end-to-end 增量成立。

## 目录

- `proposals/`：当前中文正式 Proposal；
- `decisions/`：编号化 idea、go/pivot/stop 与投稿决策；
- `notes/`：约 600 字 Idea 陈述、导师版术语表、中文工作分析、导师简报、collision 与 kill test；
- `protocol/`：文献纳入、中文文档、claim 与 seal 规则；
- `deployment/`：本地工作站、HiPerGator 资源边界、存储布局与 Slurm 模板；
- `experiments/`：runner、manifests、patch contracts、runs、predictions 与 analysis；
- `foundations/`：第一轮 10 篇、第二轮 24 篇、method-wave 新增全文，以及 MASA/Offline-RL Harness/SkillAdaptor 三个代码快照；
- `metadata/`：68 篇隔离语料的 manifests 与验证信息；
- `papers/`、`texts/`：68 篇 transport-specific PDF 与全文；
- `reading-cards/`：重点论文精读；
- `evidence-cards/`：68 份页码定位的 transport evidence maps；
- `scripts/`：检索、物化、全文抽取与项目验证脚本。

## 语料快照

原隔离语料：

- 68 条 curated records、PDF 与全文；
- 44 篇来自已验证的 1,150 篇 source corpus；
- 24 篇来自增量 arXiv 搜索与 citation chaining；
- 搜索状态：未饱和。

本轮重构新增：

- 10 篇近邻或方法论文；
- 第二轮 24 篇 exact-claim 反向检索全文；
- Harness-Bench 2.0 代码与 106 个离线 tasks；
- ToolBench-X 公开仓库可用性快照；
- 全文复核的重要直接邻居包括 Metric Freedom、AHE、Self-Harness、HarnessFix、SEAGym、LIFE-HARNESS、Harness-Bench、HASP、ContractSkill、ToolBench-X 与 The Verifier Tax。
- design-method 扩张新增 MASA、SkVM、Offline-RL Harness、Adaptive Auto-Harness、SkillAdaptor、PAR 与 UniOT 全文边界审计。

## 当前入口

请从 [START_HERE.md](START_HERE.md) 开始。

当前候选主线是 [TRACE-H Policy Transport 正式 Proposal](proposals/trace-h-policy-transport-proposal-zh.md)，当前决策是 [DR-0003B design-method pivot](decisions/0003b-trace-h-policy-transport-pivot-zh.md)。

执行前先读[本地工作站与 HiPerGator 分阶段方案](deployment/local-vs-hipergator-execution-plan-zh.md)；集群命令与模板在 [deployment/hipergator](deployment/hipergator/README.md)。

## 语言规则

从 2026-07-11 起，面向导师与研究决策的文档默认中文；历史英文 evidence cards 保留，不因语言统一而改写证据。详见 [中文文档规范](protocol/document-language-policy.md)。
